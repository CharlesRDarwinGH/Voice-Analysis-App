import sys
import wave
import pyaudio
import numpy as np
import librosa
import librosa.display
import joblib
import os
from collections import Counter
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QPushButton, QLabel, QProgressBar, QTextEdit, QApplication, QSplitter)
from PyQt5.QtCore import QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from google.cloud import language_v1
from google.cloud import speech_v1
import google.generativeai as genai
from dotenv import load_dotenv

credentials_path = "path/to/your/credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

class VoiceAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ses Analiz Uygulaması")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize clients
        self.setup_clients()

        # Initialize model attributes
        self.speaker_model = None
        self.label_encoder = None
        self.scaler = None

        self._init_ui()
        
        self.load_models()

    def setup_clients(self):
        """Google Cloud istemcilerini ayarlama"""
        self.language_client = language_v1.LanguageServiceClient()
        self.speech_client = speech_v1.SpeechClient()
        self.gemini_model = genai.GenerativeModel('gemini-pro')

    def _init_ui(self):
        """Kullanıcı arayüzü bileşenlerini başlatma"""
        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Üst kısım - Kontroller
        top_layout = QHBoxLayout()

        # Kayıt kontrolleri
        control_layout = QVBoxLayout()
        self.record_button = QPushButton("Kayıt Başlat")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet("font-size: 14px; padding: 8px;")

        self.status_label = QLabel("Hazır")
        self.status_label.setStyleSheet("font-size: 12px; color: gray;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.hide()

        control_layout.addWidget(self.record_button)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.progress_bar)
        control_layout.addStretch()

        top_layout.addLayout(control_layout)
        main_layout.addLayout(top_layout)

         # Ana splitter
        main_splitter = QSplitter()
        main_layout.addWidget(main_splitter)
        
        # Sol Taraf - Grafik Alanı
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)

        # Sağ Taraf - Sonuç Alanı
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Konuşmacı analiz sonuçları
        self.audio_results = QTextEdit()
        self.audio_results.setReadOnly(True)
        self.audio_results.setStyleSheet("font-size: 12px;")
        self.audio_results.setPlaceholderText("Ses Analiz Sonuçları")
        right_layout.addWidget(self.audio_results)
        self.audio_results.setMaximumHeight(200)
        self.audio_results.setMinimumHeight(150)

        # İçerik analiz sonuçları
        self.content_results = QTextEdit()
        self.content_results.setReadOnly(True)
        self.content_results.setStyleSheet("font-size: 12px;")
        self.content_results.setPlaceholderText("İçerik Analiz Sonuçları")
        right_layout.addWidget(self.content_results)
        self.content_results.setMinimumHeight(300)
        self.content_results.setMaximumHeight(400)

        # Splitter'a sol ve sağ widgetları ekle
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([int(self.width() * 0.8), int(self.width() * 0.2)])
        
        # Kayıt değişkenleri
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.output_file = "recorded_audio.wav"

        # Kayıt süresi için timer
        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.update_record_time)
        self.record_time = 0

        # Model eğitiminde kullanılan sınıfları sakla
        self.trained_classes = ["armağan", "nilüfer", "rana", "ömer"] 

    def load_models(self):
        try:
            # Uygulama dizinini al
            if getattr(sys, 'frozen', False):
                application_path = os.path.dirname(sys.executable)
            else:
                application_path = os.path.dirname(os.path.abspath(__file__))
                
            # Model dosyalarının varlığını kontrol et
            model_path = os.path.join(application_path, "mlp_model.pkl")
            label_encoder_path = os.path.join(application_path, "label_encoder.pkl")
            scaler_path = os.path.join(application_path, "scaler.pkl")
            
            # Dosyaların varlığını kontrol et
            if not all(os.path.exists(f) for f in [model_path, label_encoder_path, scaler_path]):
                print("Model dosyaları bulunamadı!")
                print(f"Aranan dizin: {application_path}")
                print(f"Beklenen dosyalar: {os.path.basename(model_path)}, "
                      f"{os.path.basename(label_encoder_path)}, "
                      f"{os.path.basename(scaler_path)}")
                self.status_label.setText("Model dosyaları bulunamadı!")
                self.status_label.setStyleSheet("font-size: 12px; color: red;")
                return False
            
            # Modelleri yükle
            self.speaker_model = joblib.load(model_path)
            self.label_encoder = joblib.load(label_encoder_path)
            self.scaler = joblib.load(scaler_path)

            print("Modeller başarıyla yüklendi")
            self.trained_classes = list(self.label_encoder.classes_)
            return True
                
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            print(f"Hata konumu: {application_path}")
            self.status_label.setText("Model yükleme hatası!")
            self.status_label.setStyleSheet("font-size: 12px; color: red;")
            return False

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.setText("Kayıt Durdur")
        self.status_label.setText("Kayıt yapılıyor...")
        self.status_label.setStyleSheet("font-size: 12px; color: red;")
        self.frames = []
        self.record_time = 0

        # Timer'ı başlat
        self.record_timer.start(1000)  # Her saniye güncelle

        # Progress bar'ı göster
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        # Kayıt stream'ini başlat
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )

    def update_record_time(self):
        self.record_time += 1
        self.progress_bar.setValue(min(self.record_time, 100))
        self.status_label.setText(f"Kayıt yapılıyor... ({self.record_time}s)")

    def audio_callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)

    def stop_recording(self):
        self.is_recording = False
        self.record_button.setText("Kayıt Başlat")
        self.status_label.setText("Analiz ediliyor...")
        self.status_label.setStyleSheet("font-size: 12px; color: blue;")
        self.record_timer.stop()
        self.progress_bar.hide()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Ses dosyasını kaydet
        with wave.open(self.output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))

        # Analiz yap
        self.analyze_recording()

    def analyze_text_content(self, text):
        """Basit kelime sayma ve duygu analizi ile metin analizi yapma"""
        try:
            # Duygu analizi
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT,
                language="tr"
            )
            sentiment = self.language_client.analyze_sentiment(
                request={"document": document}
            ).document_sentiment

            # Kelime sayma
            words = text.lower().split() # Tüm kelimeleri küçük harfe çevirerek say
            word_counts = Counter(words)
            most_common_keywords = word_counts.most_common(5) # İlk 5 kelimeyi seçiyoruz.

            return {
                "sentiment": {
                    "score": sentiment.score,
                    "magnitude": sentiment.magnitude
                },
                "keywords": most_common_keywords
            }
        except Exception as e:
            print(f"Metin analizi hatası: {e}")
            return None
            
    def transcribe_audio(self, audio_file_path):
        """Ses dosyasını Google Cloud Speech-to-Text API ile metne dönüştürme"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech_v1.RecognitionAudio(content=content)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=44100,  # Ses dosyası örnekleme hızı
                language_code="tr-TR",
                enable_word_time_offsets=True,
            )

            request = speech_v1.RecognizeRequest(config=config, audio=audio)
            response = self.speech_client.recognize(request=request)

            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript + " "
            return transcript.strip()

        except Exception as e:
            print(f"Speech-to-Text hatası: {e}")
            return None

    def analyze_recording(self):
        """Ses kaydını analiz etme ve sonuçları gösterme"""
        try:
            # Ses dosyasını yükle
            y, sr = librosa.load(self.output_file)

            # Grafikleri temizle
            self.figure.clear()

            # Dalga formu
            ax1 = self.figure.add_subplot(311)
            librosa.display.waveshow(y, sr=sr, ax=ax1)
            ax1.set_title('Dalga Formu')

            # Spektrogram
            ax2 = self.figure.add_subplot(312)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, y_axis='linear', ax=ax2)
            ax2.set_title('Spektrogram')

            # Konuşmacı analizi
            ax3 = self.figure.add_subplot(313)
            self.analyze_speakers(y, sr, ax3)

            self.canvas.draw()

            # Transkripsiyon ve içerik analizi
            self.perform_content_analysis()

        except Exception as e:
            self.status_label.setText(f"Analiz hatası: {e}")
            self.status_label.setStyleSheet("font-size: 12px; color: red;")

    def analyze_speakers(self, y, sr, ax):
        """Konuşmacı analizi yapma ve görselleştirme"""
        chunk_length = int(1.5 * sr)
        chunks = [y[i:i + chunk_length] for i in range(0, len(y), chunk_length)]

        speaker_times = {}
        
        # 1) Döngüde sadece tahminleri topla
        for chunk in chunks:
            if len(chunk) < chunk_length:
                continue

            # MFCC çıkarımı
            mfcc = librosa.feature.mfcc(
                y=chunk, sr=sr, n_mfcc=128, hop_length=512, n_fft=2048
            )
            # Zaman eksenini sabitleme
            if mfcc.shape[1] > 128:
                mfcc = mfcc[:, :128]
            elif mfcc.shape[1] < 128:
                pad_width = 128 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

            # Normalize + tahmin
            mfcc_flat = mfcc.flatten().reshape(1, -1)
            mfcc_normalized = self.scaler.transform(mfcc_flat)
            predictions = self.speaker_model.predict_proba(mfcc_normalized)
            predicted_class_index = np.argmax(predictions)

            # Sınıf ismi
            if self.trained_classes and predicted_class_index < len(self.trained_classes):
                speaker = self.trained_classes[predicted_class_index]
            else:
                speaker = "Bilinmeyen Kişi"

            # Süreleri biriktir
            speaker_times[speaker] = speaker_times.get(speaker, 0) + 1.5

        # 2) Pie chart'ı tek seferde çiz
        ax.clear()
        if speaker_times:
            labels = list(speaker_times.keys())
            sizes = list(speaker_times.values())
            ax.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax.set_title('Konuşmacı Dağılımı')

        # 3) Sonuçları sadece en sonda ekrana bas
        self.audio_results.clear()
        self.audio_results.append("KONUŞMACI SÜRELERİ\n")
        for speaker, time in speaker_times.items():
            self.audio_results.append(f"{speaker}: {time:.1f} saniye")

    def analyze_topics(self, text):
            """Gemini API ile konu analizi"""
            try:
                prompt = f"""Aşağıdaki Türkçe metni analiz et ve ana konularını belirle.
                Tam olarak 3 konuyu virgülle ayrılmış bir liste olarak döndür.
                Metin: {text}"""
                
                response = self.gemini_model.generate_content(prompt)
                topics = response.text.strip().split(',')
                return [topic.strip() for topic in topics[:3]]
            except Exception as e:
                print(f"Konu analizi hatası: {e}")
                return None

    def perform_content_analysis(self):
        """Transkripsiyon ve içerik analizi yapma"""
        try:
            text = self.transcribe_audio(self.output_file)
            
            if text:
                word_count = len(text.split())
                analysis_results = self.analyze_text_content(text)
                topics = self.analyze_topics(text)
                
                self.content_results.clear()
                self.content_results.append("İÇERİK ANALİZİ\n")
                self.content_results.append(f"Transkript:\n{text}\n")
                self.content_results.append(f"\nKelime Sayısı: {word_count}\n")
                
                if topics:
                    self.content_results.append("\nTespit Edilen Konular:")
                    for topic in topics:
                        self.content_results.append(f"- {topic}")
                
                if analysis_results:
                    sentiment = analysis_results["sentiment"]
                    score = sentiment['score']
                    magnitude = sentiment['magnitude']
                    score_percentage = (score + 1) / 2 * 100
                    magnitude_percentage = (magnitude / 3) * 100

                    self.content_results.append("\nDuygu Analizi:")
                    self.content_results.append(
                        f"Skor: {score_percentage:.2f}% "
                        f"(Pozitif > 50% > Negatif)"
                    )
                    self.content_results.append(
                        f"Yoğunluk: {magnitude_percentage:.2f}%\n"
                    )
                    
                    self.content_results.append("\nÖne Çıkan Kelimeler:")
                    for keyword, count in analysis_results["keywords"]:
                        self.content_results.append(f"- {keyword} ({count} kez)")

                self.status_label.setText("Analiz tamamlandı")
                self.status_label.setStyleSheet("font-size: 12px; color: green;")
            else:
                self.content_results.append("Konuşma anlaşılamadı")
                self.status_label.setText("Konuşma anlaşılamadı")
                self.status_label.setStyleSheet("font-size: 12px; color: red;")
        except Exception as e:
            self.content_results.append(f"İçerik analizi hatası: {e}")
            self.status_label.setText(f"İçerik analizi hatası: {e}")
            self.status_label.setStyleSheet("font-size: 12px; color: red;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VoiceAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
