# Voice Analysis App
Speaker Recognition, Sentiment Analysis and Subject Classification

Bu Python tabanlı proje, ses kaydındaki konuşmacıları tanır, duygusal tonları analiz eder ve konuşma içeriğini sınıflandırır.

## Gereksinimler
-Python 3.9 (önerilir) veya 3.10
Google Cloud hesabı ve etkinleştirilmiş ilgili API'ler 
-Google Cloud Natural Language 
-Google Cloud Speech-to-Text
-Gemini API)

## Kurulum
1. **Python'u yükleyin:** Python 3.9 (önerilir) veya 3.10'u indirip yükleyin. (https://www.python.org/downloads/)

### Kütüphaneler
2. Bu projeyi çalıştırmak için aşağıdaki kütüphaneleri kurmanız gerekmektedir:
```bash
pip install pyaudio numpy librosa joblib PyQt5 matplotlib google-cloud-language google-cloud-speech google-generativeai python-dotenv

3. Bu projeyi klonlayın:
   ```bash
   git clone https://github.com/kullaniciadi/projeadi.git

## API’ların kullanımı
1.	.env dosyasını yapılandırın: Proje dizinine .env adında bir dosya oluşturun ve Google Cloud API anahtarlarınızı ve diğer gerekli ortam değişkenlerini aşağıdaki gibi ekleyin:
2.	GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"  # Google Cloud kimlik bilgileri dosyanızın yolu

### 4. Uygulamayı çalıştırın:
python VoiceAnalyzer.py

## Özellikler
- **Konuşmacı Tanıma:** Ses kaydındaki farklı konuşmacıları belirler.
- **Duygu Analizi:** Konuşmanın duygusal tonunu analiz eder.
- **Konu Analizi:** Konuşmanın içeriğini sınıflandırır ve özetler.

## Lisans
Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakın.
