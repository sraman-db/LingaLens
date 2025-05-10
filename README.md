1. First Install dependencies by this code:
 `pip install -r requirements.txt`
2. Then for OCR install tesseract if not installed from `https://github.com/UB-Mannheim/tesseract/wiki`
3. After installing tesseract copy the path from your program files e.g. `C:\Program Files\Tesseract-OCR`
4. Then search Edit Environment Variables and click environment variables
5. choose Path `User variable` and click edit option then add the copied path by clicking new button
6. After that open cmd and check the tesseract version by `tesseract --version` if it shows error then check the added path in systemvariables
7. In command line write 'python app.py'
8. It will run nicely