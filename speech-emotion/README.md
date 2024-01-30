## Installation

To set up this project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://your-repository-url.git
   cd your-repository-directory
   ```

2. **Create and Activate a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install FFmpeg**
   - FFmpeg is required for audio processing. Download and install it from [FFmpeg's official site](https://ffmpeg.org/download.html).
   - Ensure that the FFmpeg binaries are in your system's PATH.

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```