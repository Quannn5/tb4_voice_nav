"""Voice and text command input for semantic Nav2 routes."""

import json
import queue
import re
import tempfile
import threading
import unicodedata
import uuid
import wave
from pathlib import Path
from typing import Any

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String

from tb4_voice_nav.reporting import append_test_result, default_results_path


_UNKNOWN_DESTINATION = re.compile(
    r'\b(?:khu vuc|phong|hanh lang)\s+([a-z0-9]+)\b'
)


def normalize_text(text: str) -> str:
    """Lowercase and remove Vietnamese diacritics for alias matching."""
    decomposed = unicodedata.normalize('NFD', text.lower())
    without_marks = ''.join(
        character
        for character in decomposed
        if unicodedata.category(character) != 'Mn'
    )
    return re.sub(r'\s+', ' ', without_marks.replace('đ', 'd')).strip()


def _route_entries(regions: list[dict[str, Any]]) -> list[dict[str, str]]:
    entries = []
    for region in regions:
        if not isinstance(region, dict) or not isinstance(region.get('name'), str):
            raise ValueError('Each semantic region must have a string name.')
        route_id = region.get('route_id')
        if not isinstance(route_id, str) or not route_id:
            route_id = re.sub(r'[^a-z0-9]+', '_', normalize_text(region['name']))
            route_id = route_id.strip('_')
        aliases = region.get('aliases', [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) for alias in aliases
        ):
            raise ValueError(
                f"Aliases for semantic region '{region['name']}' "
                'must be a list of strings.'
            )
        normalized_aliases = {
            normalize_text(alias) for alias in [region['name'], *aliases]
        }
        entries.extend(
            {'alias': alias, 'route_id': route_id, 'name': region['name']}
            for alias in normalized_aliases
            if alias
        )
    return entries


def parse_route(
    text: str,
    regions: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Extract configured destinations in spoken order and unknown mentions."""
    normalized_text = normalize_text(text)
    entries = _route_entries(regions)

    matches: list[tuple[int, int, str]] = []
    for entry in entries:
        alias_pattern = re.compile(
            rf'(?<!\w){re.escape(entry["alias"])}(?!\w)'
        )
        matches.extend(
            (match.start(), match.end(), entry['route_id'])
            for match in alias_pattern.finditer(normalized_text)
        )

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected_matches: list[tuple[int, int, str]] = []
    for candidate in matches:
        if any(
            candidate[0] < selected[1] and selected[0] < candidate[1]
            for selected in selected_matches
        ):
            continue
        selected_matches.append(candidate)

    unknown_mentions = []
    for mention in _UNKNOWN_DESTINATION.finditer(normalized_text):
        if any(
            mention.start() < selected[1] and selected[0] < mention.end()
            for selected in selected_matches
        ):
            continue
        unknown_mentions.append(mention.group(0))

    selected_matches.sort(key=lambda item: item[0])
    return (
        [match[2] for match in selected_matches],
        list(dict.fromkeys(unknown_mentions)),
    )


def _default_config_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return (
            Path(get_package_share_directory('tb4_voice_nav'))
            / 'config'
            / 'semantic_regions.yaml'
        )
    except (ImportError, LookupError):
        source_root = Path(__file__).resolve().parent.parent
        return source_root / 'config' / 'semantic_regions.yaml'


class VoiceCommander(Node):
    """Capture a Vietnamese command, validate its route, and publish JSON."""

    def __init__(self) -> None:
        super().__init__('voice_commander')
        self.declare_parameter('use_text_input', False)
        self.declare_parameter('semantic_regions_file', str(_default_config_path()))
        self.declare_parameter(
            'report_results_file',
            str(default_results_path()),
        )
        self.declare_parameter('record_seconds', 5.0)
        self.declare_parameter('stt_backend', 'google')
        self.declare_parameter('offline_fallback', 'vosk')
        self.declare_parameter('vosk_model_path', '')
        self.declare_parameter('whisper_model', 'small')

        self.publisher = self.create_publisher(
            String,
            '/voice_nav/route_command',
            10,
        )
        self.use_text_input = bool(
            self.get_parameter('use_text_input').value
        )
        self.record_seconds = float(
            self.get_parameter('record_seconds').value
        )
        self.stt_backend = str(self.get_parameter('stt_backend').value).lower()
        self.offline_fallback = str(
            self.get_parameter('offline_fallback').value
        ).lower()
        self.vosk_model_path = str(
            self.get_parameter('vosk_model_path').value
        )
        self.whisper_model_name = str(
            self.get_parameter('whisper_model').value
        )
        self.report_results_file = Path(
            str(self.get_parameter('report_results_file').value)
        ).expanduser()
        self._offline_model: Any = None

        if not 4.0 <= self.record_seconds <= 5.0:
            raise ValueError('record_seconds must be between 4.0 and 5.0.')
        if self.stt_backend not in ('google', 'vosk', 'whisper'):
            raise ValueError(
                "stt_backend must be 'google', 'vosk', or 'whisper'."
            )
        if self.offline_fallback not in ('none', 'vosk', 'whisper'):
            raise ValueError(
                "offline_fallback must be 'none', 'vosk', or 'whisper'."
            )

        config_path = Path(
            str(self.get_parameter('semantic_regions_file').value)
        ).expanduser()
        try:
            with config_path.open(encoding='utf-8') as config_file:
                config = yaml.safe_load(config_file)
            if not isinstance(config, dict) or not isinstance(
                config.get('regions'), list
            ):
                raise ValueError('Configuration must contain a regions list.')
            self.regions = config['regions']
            _route_entries(self.regions)
        except (OSError, yaml.YAMLError, ValueError) as error:
            self.get_logger().error(
                f'Cannot load semantic regions from {config_path}: {error}'
            )
            raise

        self.get_logger().info(
            f'Loaded {len(self.regions)} semantic regions from {config_path}.'
        )

    def _capture_audio(self) -> Any:
        import speech_recognition as speech

        recognizer = speech.Recognizer()
        with speech.Microphone() as microphone:
            self.get_logger().info(
                f'Recording for {self.record_seconds:.1f} seconds...'
            )
            return recognizer.record(
                microphone,
                duration=self.record_seconds,
            )

    def _transcribe_google(self, audio: Any) -> str:
        import speech_recognition as speech

        return speech.Recognizer().recognize_google(
            audio,
            language='vi-VN',
        )

    def _transcribe_offline(self, audio: Any, backend: str) -> str:
        if backend == 'vosk':
            return self._transcribe_vosk(audio)
        if backend == 'whisper':
            return self._transcribe_whisper(audio)
        raise ValueError(f'Unsupported offline STT backend: {backend}')

    def _audio_pcm(self, audio: Any) -> bytes:
        return audio.get_raw_data(convert_rate=16000, convert_width=2)

    def _transcribe_vosk(self, audio: Any) -> str:
        if not self.vosk_model_path:
            raise ValueError(
                'Set the vosk_model_path parameter to a downloaded '
                'Vietnamese Vosk model directory.'
            )
        from vosk import KaldiRecognizer, Model

        if self._offline_model is None:
            self._offline_model = Model(self.vosk_model_path)
        recognizer = KaldiRecognizer(self._offline_model, 16000)
        recognizer.AcceptWaveform(self._audio_pcm(audio))
        result = json.loads(recognizer.FinalResult())
        return str(result.get('text', '')).strip()

    def _transcribe_whisper(self, audio: Any) -> str:
        from faster_whisper import WhisperModel

        if self._offline_model is None:
            self._offline_model = WhisperModel(
                self.whisper_model_name,
                device='cpu',
                compute_type='int8',
            )
        with tempfile.TemporaryDirectory(prefix='tb4_voice_nav_') as directory:
            temporary_wav = Path(directory) / 'command.wav'
            with wave.open(str(temporary_wav), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(self._audio_pcm(audio))
            segments, _ = self._offline_model.transcribe(
                str(temporary_wav),
                language='vi',
                beam_size=5,
            )
            return ' '.join(
                segment.text.strip() for segment in segments
            ).strip()

    def _get_transcription(self) -> str:
        if self.use_text_input:
            return input('Nhập câu lệnh tiếng Việt: ').strip()

        try:
            audio = self._capture_audio()
        except (ImportError, OSError) as error:
            self.get_logger().error(
                f'Cannot access microphone or SpeechRecognition: {error}'
            )
            return ''

        try:
            if self.stt_backend == 'google':
                try:
                    return self._transcribe_google(audio)
                except Exception as error:
                    if self.offline_fallback == 'none':
                        raise
                    self.get_logger().warning(
                        f'Google vi-VN recognition failed ({error}); '
                        f'trying offline {self.offline_fallback}.'
                    )
                    return self._transcribe_offline(
                        audio,
                        self.offline_fallback,
                    )
            return self._transcribe_offline(audio, self.stt_backend)
        except Exception as error:
            self.get_logger().error(f'Speech recognition failed: {error}')
            return ''

    def _append_test_result(
        self,
        transcription: str,
        route: list[str],
        outcome: str,
        unknown_mentions: list[str] | None = None,
    ) -> None:
        region_by_id = {
            region.get('route_id'): region
            for region in self.regions
            if isinstance(region, dict)
        }
        coordinates = []
        for route_id in route:
            region = region_by_id.get(route_id)
            pose = region.get('pose') if region else None
            if isinstance(pose, list) and len(pose) >= 2:
                coordinates.append(
                    f"{route_id}: ({float(pose[0]):.2f}, {float(pose[1]):.2f})"
                )
        route_display = ' -> '.join(route)
        if unknown_mentions:
            route_display = ' -> '.join(
                [route_display, *unknown_mentions]
            ).strip(' ->')
        try:
            append_test_result(
                self.report_results_file,
                {
                    'command': transcription or '(không nhận dạng được)',
                    'transcription': transcription or '(không nhận dạng được)',
                    'route': route_display or '(không trích xuất được)',
                    'coordinates': '; '.join(coordinates) or '—',
                    'duration': '—',
                    'outcome': outcome,
                },
            )
        except OSError as error:
            self.get_logger().error(
                f'Cannot append test result to '
                f'{self.report_results_file}: {error}'
            )

    def process_transcription(self, text: str) -> bool:
        """Validate a transcription and publish its JSON route if valid."""
        if not text:
            self.get_logger().warning('No speech or text command was provided.')
            self._append_test_result(
                text,
                [],
                'Báo lỗi: không nhận dạng được câu lệnh',
            )
            return False

        self.get_logger().info(f'Recognized: {text}')
        route, unknown_mentions = parse_route(text, self.regions)
        if unknown_mentions:
            self.get_logger().error(
                'Unknown destination(s): '
                + ', '.join(unknown_mentions)
                + '. Command cancelled; nothing was published.'
            )
            self._append_test_result(
                text,
                route,
                'Báo lỗi vùng không tồn tại',
                unknown_mentions,
            )
            return False
        if not route:
            self.get_logger().error(
                'No configured destination was found; '
                'nothing was published.'
            )
            self._append_test_result(
                text,
                route,
                'Báo lỗi: không trích xuất được khu vực',
            )
            return False

        message = String()
        message.data = json.dumps(
            {
                'test_id': str(uuid.uuid4()),
                'command': text,
                'transcription': text,
                'route': route,
            },
            ensure_ascii=False,
        )
        self.publisher.publish(message)
        self.get_logger().info(
            f'Published route: {json.dumps(route, ensure_ascii=False)}'
        )
        return True


def main(args: list[str] | None = None) -> None:
    """Initialize ROS and run the voice-command window or text interface."""
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceCommander()
        if not node.use_text_input:
            _run_voice_window(node)
        else:
            print(
                'Voice commander ready: type a command for each new route. '
                'Press Ctrl+C to exit.'
            )
            while rclpy.ok():
                try:
                    transcription = node._get_transcription()
                except EOFError:
                    break
                node.process_transcription(transcription)
    except KeyboardInterrupt:
        pass
    except (ImportError, RuntimeError) as error:
        if node is not None:
            node.get_logger().error(f'Cannot start voice-command window: {error}')
        else:
            raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _run_voice_window(node: VoiceCommander) -> None:
    """Show a button-driven five-second recording and transcription window."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title('TurtleBot 4 - Điều khiển bằng giọng nói')
    root.geometry('560x300')
    root.minsize(480, 260)

    status_text = tk.StringVar(value='Nhấn nút để ghi âm trong 5 giây.')
    transcription_text = tk.StringVar(value='Văn bản nhận dạng sẽ hiện ở đây.')
    result_queue: queue.Queue[str] = queue.Queue()
    recording = False

    container = ttk.Frame(root, padding=20)
    container.pack(fill='both', expand=True)
    ttk.Label(
        container,
        text='Điều khiển TurtleBot 4',
        font=('TkDefaultFont', 16, 'bold'),
    ).pack(pady=(0, 12))

    record_button = ttk.Button(
        container,
        text=f'Ghi âm {node.record_seconds:g} giây',
    )
    record_button.pack(pady=(0, 12))
    ttk.Label(container, textvariable=status_text).pack(pady=4)
    ttk.Label(
        container,
        textvariable=transcription_text,
        wraplength=500,
        justify='center',
    ).pack(fill='x', pady=8)
    ttk.Label(
        container,
        text='Đóng cửa sổ để thoát. Câu lệnh hợp lệ sẽ được gửi tới Nav2.',
    ).pack(side='bottom', pady=(12, 0))

    def finish_recording() -> None:
        nonlocal recording
        try:
            transcription = result_queue.get_nowait()
        except queue.Empty:
            root.after(100, finish_recording)
            return

        recording = False
        record_button.configure(state='normal')
        transcription_text.set(transcription or '(Không nhận dạng được giọng nói.)')
        if transcription and node.process_transcription(transcription):
            status_text.set('Đã gửi lộ trình tới robot.')
        else:
            status_text.set('Không gửi lộ trình. Hãy xem thông báo lỗi trong terminal.')

    def start_recording() -> None:
        nonlocal recording
        if recording:
            return
        recording = True
        record_button.configure(state='disabled')
        status_text.set(
            f'Đang ghi âm {node.record_seconds:g} giây... Hãy nói điểm đến.'
        )
        transcription_text.set('')

        def capture_and_transcribe() -> None:
            result_queue.put(node._get_transcription())

        threading.Thread(
            target=capture_and_transcribe,
            name='voice-recording',
            daemon=True,
        ).start()
        root.after(100, finish_recording)

    record_button.configure(command=start_recording)

    def close_window() -> None:
        if recording:
            status_text.set('Đang ghi âm; vui lòng đợi hoàn tất rồi thoát.')
            root.after(250, close_window)
            return
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', close_window)
    root.mainloop()


if __name__ == '__main__':
    main()
