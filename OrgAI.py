from __future__ import annotations

import ctypes
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import QParallelAnimationGroup, QPropertyAnimation, QSequentialAnimationGroup, Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

APP_TITLE = "OrgAI"
APP_VERSION = "2.0"
APP_ID = "br.orgai.desktop"
APP_COPYRIGHT = '© 2026, Jeiel Lima Miranda. Todos os direitos reservados.<br><br><a href="https://miranda3000-cpu.github.io/OrgAI/">Visitar Website</a>'
BUTTON_HEIGHT = 42


def get_resource_path(relative_path: str) -> str:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base_path / relative_path)


def apply_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def load_app_icon() -> QIcon:
    for candidate in ("logo.ico", "image.png"):
        resource_path = Path(get_resource_path(candidate))
        if not resource_path.exists():
            continue
        icon = QIcon(str(resource_path))
        if not icon.isNull():
            return icon
    return QIcon()


@dataclass(frozen=True)
class PredictionInfo:
    extension: str
    confidence: float


@dataclass(frozen=True)
class FileSuggestion:
    source: Path
    extension_tag: str
    destination_folder: Path
    reason: str


class LearningNameModel:
    """Simple local learning model: multinomial Naive Bayes over filename tokens."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or (Path.home() / ".orgai_learning_model.json")
        self.token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.extension_totals: Counter[str] = Counter()
        self.total_samples = 0
        self.load()

    @staticmethod
    def tokenize(file_name: str) -> list[str]:
        stem = Path(file_name).stem.lower()
        tokens = re.split(r"[^a-z0-9]+", stem)
        return [token for token in tokens if len(token) >= 3]

    def load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            payload = json.loads(self.model_path.read_text(encoding="utf-8"))
            self.total_samples = int(payload.get("total_samples", 0))
            self.extension_totals = Counter(payload.get("extension_totals", {}))
            raw_counts = payload.get("token_counts", {})
            self.token_counts = defaultdict(Counter)
            for ext, mapping in raw_counts.items():
                self.token_counts[ext] = Counter(mapping)
        except Exception:
            self.total_samples = 0
            self.extension_totals = Counter()
            self.token_counts = defaultdict(Counter)

    def save(self) -> None:
        payload = {
            "total_samples": self.total_samples,
            "extension_totals": dict(self.extension_totals),
            "token_counts": {key: dict(value) for key, value in self.token_counts.items()},
        }
        self.model_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def learn(self, file_name: str, extension: str) -> None:
        tokens = self.tokenize(file_name)
        if not tokens:
            return
        self.total_samples += 1
        self.extension_totals[extension] += 1
        for token in tokens:
            self.token_counts[extension][token] += 1

    def predict(self, file_name: str) -> PredictionInfo | None:
        tokens = self.tokenize(file_name)
        if not tokens or self.total_samples == 0 or not self.extension_totals:
            return None

        classes = list(self.extension_totals.keys())
        vocabulary: set[str] = set()
        for ext in classes:
            vocabulary.update(self.token_counts[ext].keys())

        vocab_size = max(1, len(vocabulary))
        class_count = len(classes)
        log_scores: dict[str, float] = {}

        for ext in classes:
            prior = (self.extension_totals[ext] + 1) / (self.total_samples + class_count)
            log_prob = math.log(prior)
            token_total = sum(self.token_counts[ext].values()) + vocab_size
            for token in tokens:
                token_frequency = self.token_counts[ext].get(token, 0) + 1
                log_prob += math.log(token_frequency / token_total)
            log_scores[ext] = log_prob

        sorted_scores = sorted(log_scores.items(), key=lambda item: item[1], reverse=True)
        best_ext, best_score = sorted_scores[0]
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else best_score - 1.5

        margin = max(0.0, best_score - second_score)
        confidence = min(0.99, 0.5 + (margin / (margin + 4.0)))
        return PredictionInfo(extension=best_ext, confidence=confidence)


class AIAssistant:
    EXTENSION_ALIASES = {
        "jpeg": "jpg",
        "tiff": "tif",
        "htm": "html",
        "yml": "yaml",
    }

    EXTENSION_HINTS = {
        "PDF": "Documento de leitura identificado.",
        "DOC": "Documento de texto detectado.",
        "DOCX": "Documento de texto detectado.",
        "XLS": "Planilha identificada.",
        "XLSX": "Planilha identificada.",
        "CSV": "Tabela de dados identificada.",
        "TXT": "Arquivo de texto simples.",
        "JPG": "Imagem detectada.",
        "PNG": "Imagem detectada.",
        "GIF": "Imagem animada detectada.",
        "MP4": "Video identificado.",
        "MP3": "Audio identificado.",
        "ZIP": "Arquivo compactado detectado.",
        "RAR": "Arquivo compactado detectado.",
        "EXE": "Aplicativo executavel detectado.",
        "SEM_EXTENSAO": "Arquivo sem extensao; separado para revisao.",
    }

    def __init__(self) -> None:
        self.learning_model = LearningNameModel()

    def normalize_extension(self, source: Path) -> str:
        suffix = source.suffix.lower().lstrip(".")
        if not suffix:
            return "SEM_EXTENSAO"

        suffix = self.EXTENSION_ALIASES.get(suffix, suffix)
        normalized = "".join(ch for ch in suffix.upper() if ch.isalnum() or ch in {"_", "-"})
        if not normalized:
            return "SEM_EXTENSAO"
        return normalized[:24]

    def explain_decision(self, extension_tag: str, source: Path) -> str:
        base_hint = self.EXTENSION_HINTS.get(extension_tag, f"Extensao .{extension_tag.lower()} detectada.")
        prediction = self.learning_model.predict(source.name)
        if prediction is None:
            return f"Regra base: {base_hint}"

        confidence_text = f"{int(prediction.confidence * 100)}%"
        if prediction.extension == extension_tag:
            return f"Modelo confirma a extensao ({confidence_text}). {base_hint}"

        return (
            f"Modelo aprendeu nome parecido com .{prediction.extension.lower()} "
            f"({confidence_text}), mas extensao atual prevalece. {base_hint}"
        )

    def build_suggestions(self, folder: Path, files: Iterable[Path]) -> list[FileSuggestion]:
        suggestions: list[FileSuggestion] = []
        for source in sorted(files, key=lambda path: path.name.lower()):
            extension_tag = self.normalize_extension(source)
            destination_folder = folder / f"{extension_tag}_FILES"
            suggestions.append(
                FileSuggestion(
                    source=source,
                    extension_tag=extension_tag,
                    destination_folder=destination_folder,
                    reason=self.explain_decision(extension_tag, source),
                )
            )
        return suggestions

    def learn_batch(self, suggestions: list[FileSuggestion]) -> None:
        for suggestion in suggestions:
            self.learning_model.learn(suggestion.source.name, suggestion.extension_tag)
        self.learning_model.save()


class FileOrganizerApp(QWidget):
    def __init__(self, app_icon: QIcon):
        super().__init__()
        self.app_icon = app_icon
        self.assistant = AIAssistant()
        self.folder_path: Path | None = None
        self.suggestions: list[FileSuggestion] = []
        self.active_animations: list[QPropertyAnimation | QParallelAnimationGroup | QSequentialAnimationGroup] = []
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle(f"{APP_TITLE} - Organizador Inteligente")
        self.setFixedSize(440, 580)

        if not self.app_icon.isNull():
            self.setWindowIcon(self.app_icon)

        self.setObjectName("root")
        self.setStyleSheet(
            """
            QWidget#root {
                background-color: #f1f3f5;
                color: #212529;
                font-family: "Segoe UI";
            }
            QFrame#card {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 12px;
            }
            QLabel#appTitle {
                font-size: 24px;
                font-weight: 700;
                color: #0d6efd;
            }
            QLabel#subtitle {
                font-size: 12px;
                color: #6c757d;
            }
            QLabel#stepChip {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 5px 8px;
                font-size: 11px;
                color: #6c757d;
                background-color: #f8f9fa;
            }
            QLabel#stepChip[state="active"] {
                border: 1px solid #0d6efd;
                color: #0d6efd;
                background-color: #e7f1ff;
                font-weight: 700;
            }
            QLabel#stepChip[state="done"] {
                border: 1px solid #198754;
                color: #146c43;
                background-color: #e9f7ef;
                font-weight: 700;
            }
            QLineEdit {
                border: 1px solid #ced4da;
                border-radius: 8px;
                padding: 10px 12px;
                background-color: #ffffff;
                color: #212529;
                selection-background-color: #0d6efd;
                selection-color: #ffffff;
            }
            QLineEdit[hasValue="true"] {
                border: 2px solid #0d6efd;
                background-color: #f8fbff;
                color: #0b2e59;
                font-weight: 600;
            }
            QLineEdit:focus {
                border: 2px solid #fd7e14;
                background-color: #ffffff;
                color: #212529;
            }
            QPushButton {
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 700;
                min-height: 42px;
                max-height: 42px;
            }
            QPushButton#primary {
                background-color: #0d6efd;
                color: #ffffff;
            }
            QPushButton#primary:hover {
                background-color: #0b5ed7;
            }
            QPushButton#success {
                background-color: #198754;
                color: #ffffff;
            }
            QPushButton#success:hover {
                background-color: #157347;
            }
            QPushButton#secondary {
                background-color: #6c757d;
                color: #ffffff;
            }
            QPushButton#secondary:hover {
                background-color: #5c636a;
            }
            QPushButton#ghost {
                background-color: transparent;
                color: #0d6efd;
                border: 1px solid #0d6efd;
                min-height: 34px;
                max-height: 34px;
            }
            QPushButton#ghost:hover {
                background-color: #e9f2ff;
            }
            QPushButton[state="locked"] {
                background-color: #e9ecef;
                color: #6c757d;
            }
            QFrame#terminalCard {
                border: 1px solid #dee2e6;
                border-radius: 10px;
                background-color: #ffffff;
            }
            QLabel#terminalTitle {
                color: #6c757d;
                font-size: 11px;
                font-weight: 700;
                padding: 2px 4px;
            }
            QScrollArea#resultScroll {
                border: 1px solid #ced4da;
                border-radius: 8px;
                background-color: #ffffff;
            }
            QWidget#resultContainer {
                background-color: #ffffff;
            }
            QLabel#resultContent {
                color: #212529;
                background-color: #ffffff;
                font-size: 13px;
                padding: 8px;
            }
            QProgressBar {
                min-height: 18px;
                border: 1px solid #ced4da;
                border-radius: 8px;
                text-align: center;
                background-color: #f8f9fa;
                color: #212529;
                font-weight: 600;
            }
            QProgressBar::chunk {
                background-color: #0d6efd;
                border-radius: 7px;
            }
            """
        )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 14, 14, 14)
        card_layout.setSpacing(10)

        header_layout = QHBoxLayout()
        title_layout = QVBoxLayout()

        title = QLabel(APP_TITLE)
        title.setObjectName("appTitle")
        title.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))

        subtitle = QLabel("Fluxo com IA: selecionar -> analisar -> organizar")
        subtitle.setObjectName("subtitle")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        about_button = QPushButton("Sobre")
        about_button.setObjectName("ghost")
        about_button.clicked.connect(self.show_about_dialog)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        header_layout.addWidget(about_button)
        card_layout.addLayout(header_layout)

        step_layout = QHBoxLayout()
        self.step_select = QLabel("1 Selecionar")
        self.step_analyze = QLabel("2 Analisar IA")
        self.step_organize = QLabel("3 Organizar")
        for step in (self.step_select, self.step_analyze, self.step_organize):
            step.setObjectName("stepChip")
            step_layout.addWidget(step)
        card_layout.addLayout(step_layout)

        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        self.folder_input.setPlaceholderText("Selecione uma pasta para iniciar")
        self.folder_input.setProperty("hasValue", False)

        self.select_button = QPushButton("Selecionar pasta")
        self.select_button.setObjectName("secondary")
        self.select_button.clicked.connect(self.select_folder)

        self.analyze_button = QPushButton("Analisar com IA")
        self.analyze_button.setObjectName("primary")
        self.analyze_button.clicked.connect(self.analyze_folder)

        self.organize_button = QPushButton("Organizar agora")
        self.organize_button.setObjectName("success")
        self.organize_button.clicked.connect(self.organize_files)

        card_layout.addWidget(self.folder_input)
        card_layout.addWidget(self.select_button)
        card_layout.addWidget(self.analyze_button)
        card_layout.addWidget(self.organize_button)

        terminal_card = QFrame()
        terminal_card.setObjectName("terminalCard")
        terminal_layout = QVBoxLayout(terminal_card)
        terminal_layout.setContentsMargins(10, 10, 10, 10)
        terminal_layout.setSpacing(8)

        terminal_title = QLabel("Terminal da IA")
        terminal_title.setObjectName("terminalTitle")
        terminal_layout.addWidget(terminal_title)

        self.result_scroll = QScrollArea()
        self.result_scroll.setObjectName("resultScroll")
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setMinimumHeight(200)

        result_container = QWidget()
        result_container.setObjectName("resultContainer")
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(6, 6, 6, 6)

        self.result_content = QLabel(
            "Resultado da IA aparecera aqui.\n\n"
            "Fluxo:\n"
            "1) Selecionar pasta\n"
            "2) Analisar com IA\n"
            "3) Organizar"
        )
        self.result_content.setObjectName("resultContent")
        self.result_content.setWordWrap(True)
        self.result_content.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        result_layout.addWidget(self.result_content)
        result_layout.addStretch()
        self.result_scroll.setWidget(result_container)
        terminal_layout.addWidget(self.result_scroll)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        terminal_layout.addWidget(self.progress_bar)

        card_layout.addWidget(terminal_card)
        root_layout.addWidget(card)

        self._hide_button_for_flow(self.analyze_button)
        self._hide_button_for_flow(self.organize_button)
        self._set_stage(1)

    def _register_animation(self, animation: QPropertyAnimation | QParallelAnimationGroup | QSequentialAnimationGroup) -> None:
        def _cleanup() -> None:
            if animation in self.active_animations:
                self.active_animations.remove(animation)

        animation.finished.connect(_cleanup)
        self.active_animations.append(animation)
        animation.start()

    def _hide_button_for_flow(self, button: QPushButton) -> None:
        button.setVisible(False)
        button.setEnabled(False)
        button.setMinimumHeight(0)
        button.setMaximumHeight(0)
        button.setProperty("state", "locked")
        self._refresh_widget_style(button)

    def _reveal_with_pulse(self, button: QPushButton) -> None:
        button.setVisible(True)
        button.setEnabled(True)
        button.setProperty("state", "ready")
        self._refresh_widget_style(button)

        grow = QParallelAnimationGroup(self)

        min_grow = QPropertyAnimation(button, b"minimumHeight", self)
        min_grow.setDuration(170)
        min_grow.setStartValue(0)
        min_grow.setEndValue(BUTTON_HEIGHT)

        max_grow = QPropertyAnimation(button, b"maximumHeight", self)
        max_grow.setDuration(170)
        max_grow.setStartValue(0)
        max_grow.setEndValue(BUTTON_HEIGHT)

        grow.addAnimation(min_grow)
        grow.addAnimation(max_grow)

        pulse = self._build_pulse_animation(button)

        sequence = QSequentialAnimationGroup(self)
        sequence.addAnimation(grow)
        sequence.addAnimation(pulse)
        self._register_animation(sequence)

    def _build_pulse_animation(self, button: QPushButton) -> QSequentialAnimationGroup:
        pulse_up = QParallelAnimationGroup(self)
        pulse_down = QParallelAnimationGroup(self)

        min_up = QPropertyAnimation(button, b"minimumHeight", self)
        min_up.setDuration(120)
        min_up.setStartValue(BUTTON_HEIGHT)
        min_up.setEndValue(BUTTON_HEIGHT + 4)

        max_up = QPropertyAnimation(button, b"maximumHeight", self)
        max_up.setDuration(120)
        max_up.setStartValue(BUTTON_HEIGHT)
        max_up.setEndValue(BUTTON_HEIGHT + 4)

        min_down = QPropertyAnimation(button, b"minimumHeight", self)
        min_down.setDuration(120)
        min_down.setStartValue(BUTTON_HEIGHT + 4)
        min_down.setEndValue(BUTTON_HEIGHT)

        max_down = QPropertyAnimation(button, b"maximumHeight", self)
        max_down.setDuration(120)
        max_down.setStartValue(BUTTON_HEIGHT + 4)
        max_down.setEndValue(BUTTON_HEIGHT)

        pulse_up.addAnimation(min_up)
        pulse_up.addAnimation(max_up)
        pulse_down.addAnimation(min_down)
        pulse_down.addAnimation(max_down)

        sequence = QSequentialAnimationGroup(self)
        sequence.addAnimation(pulse_up)
        sequence.addAnimation(pulse_down)
        return sequence

    def _pulse_button(self, button: QPushButton) -> None:
        pulse = self._build_pulse_animation(button)
        self._register_animation(pulse)

    def _set_stage(self, stage: int) -> None:
        chips = [self.step_select, self.step_analyze, self.step_organize]
        for index, chip in enumerate(chips, start=1):
            if index < stage:
                state = "done"
            elif index == stage:
                state = "active"
            else:
                state = "idle"
            chip.setProperty("state", state)
            self._refresh_widget_style(chip)

    @staticmethod
    def _refresh_widget_style(widget: QWidget) -> None:
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)
        widget.update()

    def _set_result_text(self, text: str) -> None:
        self.result_content.setText(text)
        scrollbar = self.result_scroll.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(0)

    def show_modal(self, title: str, message: str, icon: QMessageBox.Icon) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setIcon(icon)
        if not self.app_icon.isNull():
            dialog.setWindowIcon(self.app_icon)

        dialog.setStyleSheet(
            """
            QMessageBox { background-color: #ffffff; font-size: 12px; }
            QMessageBox QLabel { color: #0f172a; min-width: 220px; }
            QMessageBox QPushButton {
                min-width: 72px;
                padding: 6px 10px;
                background-color: #0d6efd;
                color: #ffffff;
                border-radius: 8px;
            }
            """
        )
        dialog.setText(message)
        dialog.exec()

    def show_about_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Sobre")
        dialog.setModal(True)
        dialog.setFixedSize(280, 120)
        dialog.setWindowFlag(Qt.WindowType.MSWindowsFixedSizeDialogHint, True)

        if not self.app_icon.isNull():
            dialog.setWindowIcon(self.app_icon)

        dialog.setStyleSheet(
            """
            QDialog {
                background-color: #ffffff;
            }
            QLabel#aboutInfo {
                color: #0f172a;
                font-size: 12px;
                padding: 8px;
            }
            """
        )

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)

        info = QLabel(f"{APP_TITLE} v{APP_VERSION}\nOrganizador por extensao com IA local.\n\n {APP_COPYRIGHT}", dialog)
        info.setObjectName("aboutInfo")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setOpenExternalLinks(True)
        info.setTextFormat(Qt.TextFormat.RichText)
        info.setWordWrap(True)

        layout.addWidget(info)
        dialog.exec()

    def select_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Selecione a pasta")
        if not selected:
            return

        self.folder_path = Path(selected)
        self.folder_input.setText(str(self.folder_path))
        self.folder_input.setProperty("hasValue", True)
        self._refresh_widget_style(self.folder_input)
        self.suggestions = []

        self._set_stage(2)
        self._reveal_with_pulse(self.analyze_button)
        self._hide_button_for_flow(self.organize_button)

        self._set_result_text(
            "Pasta selecionada com sucesso.\n"
            "Proximo passo: clique em 'Analisar com IA'."
        )

    def list_root_files(self) -> list[Path]:
        if self.folder_path is None:
            return []
        return sorted(
            [path for path in self.folder_path.iterdir() if path.is_file()],
            key=lambda path: path.name.lower(),
        )

    def analyze_folder(self) -> None:
        if self.folder_path is None:
            self.show_user_error("Escolha uma pasta antes de analisar.")
            return

        try:
            files = self.list_root_files()
            if not files:
                self.suggestions = []
                self._hide_button_for_flow(self.organize_button)
                self._set_result_text("Nenhum arquivo encontrado no nivel principal da pasta selecionada.")
                return

            self.suggestions = self.assistant.build_suggestions(self.folder_path, files)
            self.populate_preview(self.suggestions)

            self._set_stage(3)
            self._reveal_with_pulse(self.organize_button)

        except Exception as error:
            self.show_user_error(
                "Nao foi possivel analisar a pasta agora. Tente novamente.",
                technical_error=error,
            )

    def populate_preview(self, suggestions: list[FileSuggestion]) -> None:
        extension_count = len({suggestion.extension_tag for suggestion in suggestions})
        lines = [
            f"IA analisou {len(suggestions)} arquivo(s) em {extension_count} extensao(oes).",
            "",
            "Plano de organizacao:",
        ]

        for suggestion in suggestions:
            lines.extend(
                [
                    f"- {suggestion.source.name}",
                    f"  -> {suggestion.destination_folder.name}",
                    f"  IA: {suggestion.reason}",
                    "",
                ]
            )

        self._set_result_text("\n".join(lines).strip())

    def organize_files(self) -> None:
        if self.folder_path is None:
            self.show_user_error("Escolha uma pasta antes de organizar.")
            return

        if not self.suggestions:
            self.analyze_folder()
            if not self.suggestions:
                return

        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.suggestions))
        self.progress_bar.setValue(0)

        batch_snapshot = list(self.suggestions)

        try:
            for index, suggestion in enumerate(batch_snapshot, start=1):
                suggestion.destination_folder.mkdir(parents=True, exist_ok=True)
                destination_file = self.resolve_destination_collision(
                    suggestion.destination_folder,
                    suggestion.source.name,
                )
                shutil.move(str(suggestion.source), str(destination_file))
                self.progress_bar.setValue(index)
                QApplication.processEvents()

            self.assistant.learn_batch(batch_snapshot)

            moved_count = len(batch_snapshot)
            self.suggestions = []
            self._set_stage(1)
            self._hide_button_for_flow(self.organize_button)
            self._pulse_button(self.select_button)

            self._set_result_text(
                f"Organizacao concluida com sucesso. {moved_count} arquivo(s) movido(s).\n"
                "A IA local aprendeu com este lote e estara mais precisa nas proximas analises."
            )

        except PermissionError as error:
            self.show_user_error(
                "Sem permissao para mover alguns arquivos. Feche os arquivos abertos e tente novamente.",
                technical_error=error,
            )
        except FileNotFoundError as error:
            self.show_user_error(
                "Um arquivo mudou de lugar durante a organizacao. Analise novamente e tente de novo.",
                technical_error=error,
            )
        except OSError as error:
            self.show_user_error(
                "Nao foi possivel concluir agora. Verifique espaco e permissao da pasta.",
                technical_error=error,
            )
        except Exception as error:
            self.show_user_error(
                "Erro inesperado durante a organizacao. Tente novamente em instantes.",
                technical_error=error,
            )
        finally:
            self.progress_bar.setVisible(False)

    @staticmethod
    def resolve_destination_collision(destination_folder: Path, file_name: str) -> Path:
        candidate = destination_folder / file_name
        if not candidate.exists():
            return candidate

        stem = Path(file_name).stem
        suffix = Path(file_name).suffix
        index = 1

        while True:
            candidate = destination_folder / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def show_user_error(self, friendly_message: str, technical_error: Exception | None = None) -> None:
        self._set_result_text(f"Erro: {friendly_message}")
        self.show_modal("Nao foi possivel concluir", friendly_message, QMessageBox.Icon.Warning)

        if technical_error is not None:
            print(f"[DEBUG] {technical_error}")


def main() -> int:
    try:
        apply_windows_app_id()

        app = QApplication(sys.argv)
        app.setApplicationName(APP_TITLE)

        app_icon = load_app_icon()
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)

        window = FileOrganizerApp(app_icon)
        window.show()

        return app.exec()
    except Exception as error:
        print(f"[FATAL] {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
