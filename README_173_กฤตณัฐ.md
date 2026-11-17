# CPE312_Model-as-a-Service-Flask-Docker

## 1. โครงสร้างโฟลเดอร์ของโปรเจ็ค
```
ml_service/
├─ app.py
├─ train_model.py
├─ requirements.txt
├─ Dockerfile
├─ templates/
│ └─ main.html
└─ model/
    └─ model.pkl
```

### 2.ไฟล์: `train_model.py`

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib, os

data = fetch_california_housing(as_frame=False)
X, y = data.data, data.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(Xtr, ytr)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("บันทึกโมเดลเรียบร้อยที่ model/model.pkl")
```
---

### ３.ไฟล์: `app.py`

```python
from flask import Flask, request, jsonify, render_template
import joblib, numpy as np, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"]

app = Flask(__name__, template_folder="templates")

@app.get("/health")
def health():
    return {"ok": True, "features": FEATURES}

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    x = data.get("features")
    if not x or len(x) != len(FEATURES):
        return jsonify(error=f"Expected 8 features: {FEATURES}"), 400
    X = np.array([x], dtype=float)
    yhat = model.predict(X).tolist()[0]
    return jsonify(prediction=yhat)

@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return render_template("main.html")
    vals = [float(request.form.get(f)) for f in FEATURES]
    pred = model.predict(np.array([vals])).tolist()[0]
    return render_template("main.html", prediction=pred)

if __name__ == "__main__":
    app.run(debug=True)
```

### 4.ไฟล์: `templates/main.html`

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>California Housing — ML Predictor</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    body { max-width: 920px; margin: 24px auto; padding: 0 16px; }
    h1 { margin-bottom: 4px; }
    .sub { color: #555; margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 14px 20px; }
    label { font-weight: 600; display:block; margin-bottom: 6px; }
    .help { font-size: 12px; color: #666; margin-top: 4px; }
    input[type="number"] { width: 100%; padding: 10px 12px; border: 1px solid #ccc; border-radius: 10px; }
    .actions { margin-top: 16px; display:flex; gap: 10px; flex-wrap: wrap; }
    button { padding: 10px 16px; border: 0; border-radius: 10px; cursor: pointer; }
    .primary { background: #2563eb; color: white; }
    .ghost { background: #f1f5f9; }
    .result { margin-top: 18px; padding: 14px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; }
    .err { color: #b91c1c; }
    .muted { color:#64748b; font-size:12px; }
    footer { margin-top: 24px; color:#64748b; font-size:12px; }
  </style>
</head>
<body>
  <h1>California Housing — ML Predictor</h1>
  <p class="sub">Enter values and click <b>Predict</b>. The model estimates median house value (in $100,000s).</p>

  <form id="form" class="grid" autocomplete="off">
    <div>
      <label for="MedInc">MedInc (Median income, $10k units)</label>
      <input id="MedInc" name="MedInc" type="number" step="0.1" min="0" max="25" placeholder="e.g., 8.3" required />
      <div class="help">Typical 0–15 (e.g., 8.3 means $83,000).</div>
    </div>

    <div>
      <label for="HouseAge">HouseAge (years)</label>
      <input id="HouseAge" name="HouseAge" type="number" step="1" min="1" max="100" placeholder="e.g., 25" required />
      <div class="help">Average age of houses in the block.</div>
    </div>

    <div>
      <label for="AveRooms">AveRooms (avg rooms per dwelling)</label>
      <input id="AveRooms" name="AveRooms" type="number" step="0.1" min="1" max="15" placeholder="e.g., 6.0" required />
      <div class="help">Total rooms / total households.</div>
    </div>

    <div>
      <label for="AveBedrms">AveBedrms (avg bedrooms per dwelling)</label>
      <input id="AveBedrms" name="AveBedrms" type="number" step="0.1" min="0.5" max="6" placeholder="e.g., 1.0" required />
      <div class="help">Total bedrooms / total households.</div>
    </div>

    <div>
      <label for="Population">Population (people)</label>
      <input id="Population" name="Population" type="number" step="1" min="1" max="50000" placeholder="e.g., 1200" required />
      <div class="help">Block group population.</div>
    </div>

    <div>
      <label for="AveOccup">AveOccup (avg occupants per household)</label>
      <input id="AveOccup" name="AveOccup" type="number" step="0.1" min="0.5" max="10" placeholder="e.g., 3.0" required />
      <div class="help">Population / households.</div>
    </div>

    <div>
      <label for="Latitude">Latitude</label>
      <input id="Latitude" name="Latitude" type="number" step="0.01" min="32" max="42" placeholder="e.g., 34.20" required />
      <div class="help">Approx. California 32–42.</div>
    </div>

    <div>
      <label for="Longitude">Longitude</label>
      <input id="Longitude" name="Longitude" type="number" step="0.01" min="-125" max="-114" placeholder="e.g., -118.30" required />
      <div class="help">Approx. California −125 to −114.</div>
    </div>

    <div class="actions">
      <button type="button" class="ghost" id="sample">Use sample values</button>
      <button type="submit" class="primary">Predict</button>
      <button type="reset" class="ghost">Reset</button>
    </div>
  </form>

  <div id="output" class="result" style="display:none;"></div>

  <footer>
    <div class="muted">
      Feature order sent to the API: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    </div>
  </footer>

  <script>
    const FEATURES = ["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude"];
    const sample = { MedInc: 8.3, HouseAge: 25, AveRooms: 6.0, AveBedrms: 1.0, Population: 1200, AveOccup: 3.0, Latitude: 34.2, Longitude: -118.3 };

    document.getElementById('sample').addEventListener('click', () => {
      for (const [k,v] of Object.entries(sample)) document.getElementById(k).value = v;
    });

    document.getElementById('form').addEventListener('submit', async (e) => {
      e.preventDefault();
      const vals = [];
      try {
        for (const f of FEATURES) {
          const el = document.getElementById(f);
          if (!el || el.value === "") throw new Error(`Missing value: ${f}`);
          vals.push(parseFloat(el.value));
        }
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ features: vals })
        });
        const data = await res.json();
        const out = document.getElementById('output');
        out.style.display = 'block';
        if (!res.ok) {
          out.innerHTML = `<div class="err"><b>Error:</b> ${data.error || 'Request failed'}</div>`;
          return;
        }
        const y = Number(data.prediction);
        out.innerHTML = `<b>Prediction:</b> ${isFinite(y) ? y.toFixed(3) : data.prediction}
          <div class="muted">Inputs: [${vals.join(', ')}]</div>`;
      } catch (err) {
        const out = document.getElementById('output');
        out.style.display = 'block';
        out.innerHTML = `<div class="err"><b>Validation error:</b> ${err.message}</div>`;
      }
    });
  </script>
</body>
</html>

### 5.ไฟล์: `requirements.txt`

```
flask==3.1.3
gunicorn==26.2.0
numpy==2.4.6
scikit-learn==1.9.1
joblib==1.6.0
pandas==3.1.1
```

### 6.ไฟล์: `Dockerfile`

```
# Base image ที่ใช้ Python 3.11 แบบขนาดเล็ก
FROM python:3.11-slim

# สร้างโฟลเดอร์ทำงานภายใน container
WORKDIR /app

# คัดลอก requirements.txt และติดตั้ง dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจกต์ทั้งหมดเข้า container
COPY . .

# สร้างโมเดล (train model) ตอน build image
RUN python train_model.py

# เปิด port 5000 สำหรับ Flask
EXPOSE 5000

# สั่งให้รัน Flask app ผ่าน Gunicorn (production server)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]

```
---

### 7.ไฟล์: `Dockerfile`

```
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib, os

data = fetch_california_housing(as_frame=False)
X, y = data.data, data.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(Xtr, ytr)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("บันทึกโมเดลเรียบร้อยที่ model/model.pkl")
```

### 8.ไฟล์: README

```
# ML Deployment Lab: Model as a Service with Flask and Docker

> โปรเจกต์ตัวอย่างสำหรับรายวิชา CPE312 Data Science\
> สาธิตการนำ Machine Learning Model มาพัฒนาเป็นบริการ (Model as a Service:
> MaaS) ด้วย Flask API และ Docker

## 1. Project Overview

โปรเจกต์นี้ใช้ชุดข้อมูล **California Housing** เพื่อฝึกโมเดล **Random Forest
Regression** สำหรับประมาณค่ามัธยฐานของมูลค่าบ้านในพื้นที่ จากนั้นบันทึกโมเดลด้วย Joblib
และสร้าง Flask Application เพื่อรับข้อมูลจากผู้ใช้และส่งผลการทำนายกลับในรูปแบบ JSON
หรือแสดงผลผ่านหน้าเว็บไซต์

ระบบสามารถทำงานได้ 2 รูปแบบ

1.  **Local mode:** รัน Flask ด้วย Python ในเครื่อง
2.  **Docker mode:** สร้าง Docker Image และรัน Application ภายใน Container
    โดยใช้ Gunicorn

> หมายเหตุ: ผลลัพธ์เป็นค่าประมาณจากโมเดลเพื่อการศึกษา
> ไม่ใช่ราคาประเมินอสังหาริมทรัพย์อย่างเป็นทางการ

## 2. Technologies

-   Python 3.11
-   Flask --- Web Application และ REST API
-   scikit-learn --- Dataset และ Random Forest Regression
-   NumPy --- จัดการข้อมูลตัวเลข
-   Joblib --- บันทึกและโหลดโมเดล
-   Pandas --- เครื่องมือจัดการข้อมูล
-   Gunicorn --- WSGI server สำหรับรัน Flask ใน Container
-   Docker --- สร้าง Image และรัน Container
-   HTML, CSS และ JavaScript --- ส่วนติดต่อผู้ใช้

## 3. Dataset and Features

ใช้ California Housing Dataset จาก
`sklearn.datasets.fetch_california_housing`

โมเดลรับข้อมูลเข้า 8 Features ตามลำดับต่อไปนี้

  -------------------------------------------------------------------------------
                          ลำดับ Feature               ความหมาย
  ---------------------------- --------------------- ----------------------------
                             1 `MedInc`              รายได้มัธยฐานของครัวเรือนในพื้นที่
                                                     (หน่วยตาม Dataset: 10,000
                                                     ดอลลาร์สหรัฐ)

                             2 `HouseAge`            อายุบ้านมัธยฐานในพื้นที่

                             3 `AveRooms`            จำนวนห้องเฉลี่ยต่อครัวเรือน

                             4 `AveBedrms`           จำนวนห้องนอนเฉลี่ยต่อครัวเรือน

                             5 `Population`          จำนวนประชากรในพื้นที่

                             6 `AveOccup`            จำนวนผู้อยู่อาศัยเฉลี่ยต่อครัวเรือน

                             7 `Latitude`            ละติจูด

                             8 `Longitude`           ลองจิจูด
  -------------------------------------------------------------------------------

**Target:** `MedHouseVal` --- ค่ามัธยฐานของมูลค่าบ้าน หน่วยเป็น 100,000
ดอลลาร์สหรัฐ ตามนิยามของ Dataset

## 4. Project Structure

``` text
ml_service/
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── templates/
│   └── main.html
└── model/
    └── model.pkl       # สร้างจาก train_model.py; ไม่จำเป็นต้อง commit หาก Docker ฝึกโมเดลตอน build
```

## 5. Requirements

ติดตั้งโปรแกรมต่อไปนี้ก่อนเริ่มใช้งาน

-   Python 3.11
-   Visual Studio Code (หรือ Editor อื่น)
-   Git
-   Docker Desktop (สำหรับการรันด้วย Docker)

ตรวจสอบเวอร์ชัน Python และ Docker ได้ด้วยคำสั่ง

``` powershell
python --version
docker --version
```

## 6. Run Locally (Windows PowerShell)

เปิด PowerShell ในโฟลเดอร์โปรเจกต์

### 6.1 Create and activate virtual environment

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

หาก PowerShell ป้องกันการรันสคริปต์ ให้ใช้คำสั่งต่อไปนี้ในหน้าต่าง Terminal ปัจจุบัน
แล้วลอง Activate อีกครั้ง

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

เมื่อเปิดใช้งานสำเร็จ จะเห็น `(.venv)` นำหน้าบรรทัดคำสั่ง

### 6.2 Install dependencies

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6.3 Train the model

``` powershell
python train_model.py
```

สคริปต์จะดาวน์โหลด Dataset (ต้องเชื่อมต่ออินเทอร์เน็ตในครั้งแรก) แบ่งข้อมูลเป็นชุด Train
และ Test ฝึก Random Forest Regression ประเมินค่า R² และบันทึกโมเดลไว้ที่
`model/model.pkl`

### 6.4 Start Flask

``` powershell
python app.py
```

เปิดเว็บเบราว์เซอร์ที่:

-   http://127.0.0.1:5000/

หยุดเซิร์ฟเวอร์ด้วย `Ctrl + C`

## 7. API Endpoints

### `GET /health`

ตรวจสอบสถานะ Application และรายชื่อ Features

ตัวอย่าง:

``` powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/health -Method Get
```

### `POST /predict`

รับ JSON ที่มี `features` เป็น Array ตัวเลข 8 ค่า โดยต้องเรียงตามลำดับ Features
ในหัวข้อ Dataset and Features

ตัวอย่าง PowerShell:

``` powershell
$body = @{
    features = @(8.3, 25.0, 6.0, 1.0, 1200, 3.0, 34.2, -118.3)
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri http://127.0.0.1:5000/predict `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

ตัวอย่างรูปแบบ Response:

``` json
{
  "prediction": 4.0
}
```

ตัวเลขข้างต้นเป็นเพียงตัวอย่างรูปแบบการตอบกลับ ค่าจริงขึ้นอยู่กับโมเดลที่ฝึกและข้อมูลที่ส่งเข้าไป

ค่า `prediction` มีหน่วยเป็น 100,000 ดอลลาร์สหรัฐ หากต้องการแสดงเป็นดอลลาร์สหรัฐ
ให้คูณค่าที่ได้ด้วย 100,000

## 8. Run with Docker

ตรวจสอบก่อนว่า Docker Desktop เปิดใช้งานอยู่ และ PowerShell อยู่ในโฟลเดอร์เดียวกับ
`Dockerfile` และ `requirements.txt`

### 8.1 Build Docker Image

``` powershell
docker build -t ds-ml-service .
```

ใน Dockerfile ของโปรเจกต์นี้มีขั้นตอนติดตั้ง dependencies และเรียก
`train_model.py` ระหว่างการ Build ดังนั้นการ Build
ครั้งแรกต้องใช้อินเทอร์เน็ตเพื่อดาวน์โหลด Base Image และ Dataset

ตรวจสอบ Image:

``` powershell
docker images
```

### 8.2 Run Container

``` powershell
docker run --rm -p 5000:5000 ds-ml-service
```

เปิดเว็บเบราว์เซอร์ที่:

-   http://127.0.0.1:5000/
-   http://127.0.0.1:5000/health

หยุด Container ด้วย `Ctrl + C`

หาก Port 5000 ถูกใช้งานอยู่ สามารถเปลี่ยน Port ฝั่งเครื่อง Host ได้ เช่น:

``` powershell
docker run --rm -p 8080:5000 ds-ml-service
```

แล้วเปิด http://127.0.0.1:8080/

## 9. Troubleshooting

  ----------------------------------------------------------------------------------------------
  อาการ                                            แนวทางตรวจสอบ/แก้ไข
  ------------------------------------------------ ---------------------------------------------
  `ModuleNotFoundError: No module named 'flask'`   เปิดใช้งาน `.venv` และติดตั้งด้วย
                                                   `python -m pip install -r requirements.txt`
                                                   ตรวจสอบ Python Interpreter ใน VS Code ให้เป็น
                                                   `.venv\Scripts\python.exe`

  `requirements.txt: not found` ระหว่าง Docker      ตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกับ Dockerfile
  Build                                            และไม่ได้ถูกระบุไว้ใน `.dockerignore`; รัน `dir`
                                                   และ `Test-Path .\requirements.txt`

  `model/model.pkl` ไม่พบ                           รัน `python train_model.py` ก่อนเริ่ม Flask ใน
                                                   Local mode

  `TemplateNotFound: main.html`                    ตรวจสอบว่าไฟล์อยู่ที่ `templates/main.html`

  Docker ติดต่อ Engine ไม่ได้                          เปิด Docker Desktop และรอให้ Engine พร้อมใช้งาน

  Port 5000 ถูกใช้งาน                                ใช้ `-p 8080:5000` แล้วเปิด
                                                   `http://127.0.0.1:8080/`

  Docker หา Image `ds-ml-service` ไม่พบ             Build Image ให้สำเร็จก่อนด้วย
                                                   `docker build -t ds-ml-service .`
  ----------------------------------------------------------------------------------------------

## 10. Notes for Development and Submission

ก่อนส่งงานควรตรวจสอบว่า:

-   [ ] `train_model.py` ทำงานและสร้าง `model/model.pkl` ได้
-   [ ] `app.py` เปิดหน้าเว็บไซต์ได้
-   [ ] `GET /health` ตอบกลับสถานะระบบ
-   [ ] `POST /predict` รับ Features 8 ค่าและส่ง Prediction กลับ
-   [ ] Docker Image Build สำเร็จ
-   [ ] Docker Container เปิดเว็บไซต์และ API ได้
-   [ ] `README.md` และ Source Code อยู่ใน Repository
-   [ ] ไม่มีไฟล์ลับ เช่น `.env`, รหัสผ่าน หรือ Token ถูก commit
-   [ ] เพิ่มภาพหน้าจอผลการทดสอบ หากผู้สอนกำหนดให้ส่ง

### Git commands

หาก Clone Repository ของรายวิชามาแล้ว ให้ทำงานภายในโฟลเดอร์ Repository นั้น
จากนั้นตรวจสอบและบันทึกการเปลี่ยนแปลง:

``` powershell
git status
git add .
git commit -m "Complete ML deployment lab"
git push
```

ตรวจสอบ Branch และ Remote ด้วย `git branch` และ `git remote -v` ก่อน Push
หาก Repository กำหนด Branch หรือขั้นตอนส่งงานไว้ต่างจากนี้ ให้ทำตามคำสั่งของผู้สอน

## 11. Author

-   Name: **\[Your Name\]**
-   Course: **CPE312 Data Science**
-   Project: **ML Deployment Lab --- Model as a Service**

------------------------------------------------------------------------

**Educational project:** พัฒนาขึ้นเพื่อศึกษา Machine Learning Model Training,
Flask API และ Docker-based Deployment
```
