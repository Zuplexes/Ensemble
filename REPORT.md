# รายงาน: Decision Tree vs Bagging vs Boosting (เขียนเอง) บน Iris

## วิธีติดตั้งและรัน

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                          # 16 tests
python main.py                  # รันทุกการทดลอง → results/
```

ทดสอบแล้วบน Python 3.12.0 (Windows 11)

> ค่าเวลาระดับ ms ไวต่อโหลดของเครื่องมาก ควรปิดโปรแกรมหนักๆ (เกม เบราว์เซอร์หลายแท็บ) ก่อนรัน ถ้ารันตอนเครื่องมีโหลด เวลาจะสูงขึ้น 2–3 เท่าและ std กว้างผิดปกติ แต่ accuracy ไม่เปลี่ยน

## 1. ข้อมูลและการแบ่ง

- `data/Iris.csv`: 150 แถว, 4 features, 3 classes (50 ต่อคลาส) ตัดคอลัมน์ `Id` ทิ้ง แปลง label เป็น 0/1/2
- Stratified split 80/20, `random_state=42` ได้ **train 120 (40/40/40), test 30 (10/10/10)** บันทึก index ไว้ที่ `data/split_indices.npz` และทุกโมเดลโหลดชุดเดียวกันนี้
- StandardScaler fit บน train เท่านั้น (tree ไม่ได้รับผลจาก scaling อยู่แล้ว)
- เลือก hyperparameter ด้วย **stratified 5-fold CV บน train เท่านั้น** แล้วประเมินบน test ครั้งเดียวต่อโมเดลที่เลือกได้ ถ้า CV เสมอกันจะเลือกโมเดลที่เล็กกว่า (depth ตื้นกว่า / estimators น้อยกว่า)

## 2. วิธีการ

| ไฟล์ | สิ่งที่ทำ |
|---|---|
| `tree.py` | CART, Gini ถ่วงด้วย `sample_weight`, หา threshold แบบ vectorized (argsort + cumsum ของ class weight ทุกจุดตัด), threshold = จุดกึ่งกลาง, ถ้าเสมอกันเลือกจุดแรกเสมอ (deterministic) แบ่งทุก node ที่ไม่บริสุทธิ์แม้ gain = 0 เหมือน sklearn (จึงแก้ XOR ได้) `DecisionStump` = depth 1 |
| `bagging.py` | bootstrap (สุ่มแบบคืนที่ ขนาด n) ด้วย `np.random.default_rng(seed)`, majority vote ด้วย `np.bincount` ครั้งเดียวสำหรับทุก sample, บันทึก `fit_time` / `predict_time` ด้วย `time.perf_counter()` |
| `boosting.py` | AdaBoost **SAMME**: `w=1/N`, `alpha = lr·(ln((1-err)/err) + ln(K-1))`, `w *= exp(alpha·miss)` แล้ว normalize, ถ้า `err ≥ 1-1/K` หยุด, ถ้า `err = 0` ใช้ epsilon แล้วหยุด, predict ด้วย vote ถ่วงน้ำหนัก alpha, base learner = stump |

สามไฟล์นี้ import แค่ `numpy`, `math`, `time`, `typing` (ตรวจด้วย grep แล้ว ไม่มี sklearn/scipy/pandas)

**n_estimators** คือจำนวนโมเดลย่อย (base learner) ใน ensemble
- Bagging: จำนวน tree ที่ train แยกกันบน bootstrap ของตัวเอง แล้วโหวตเสียงข้างมาก
- Boosting: จำนวนรอบ แต่ละรอบเพิ่ม stump 1 ตัวที่เน้นตัวอย่างที่รอบก่อนทายผิด แล้วโหวตถ่วงน้ำหนักด้วย alpha

**การจับเวลา:** แต่ละ config fit + predict 10 รอบ (seed 0–9) รายงาน mean ± std ส่วน Tree และ Boosting ไม่มีการสุ่ม จึงได้ accuracy เท่ากันทุก seed

## 3. ผลลัพธ์

### ตารางสรุป (`results/summary.md`)

| Model | CV Acc (train) | Test Acc | Test Acc (10 seeds) | Macro F1 | Train time (mean±std) | Predict time (mean±std) |
|---|---|---|---|---|---|---|
| Decision Tree (max_depth=3) | 0.9500 | 0.9667 | 0.9667 ± 0.0000 | 0.9666 | 0.407 ± 0.049 ms | 0.024 ± 0.028 ms |
| Bagging (n=5) | 0.9583 | 0.9333 | 0.9233 ± 0.0300 | 0.9333 | 2.571 ± 0.145 ms | 0.092 ± 0.005 ms |
| Boosting SAMME (n=10, lr=1.0) | 0.9667 | 0.9333 | 0.9333 ± 0.0000 | 0.9333 | 1.260 ± 0.065 ms | 0.050 ± 0.003 ms |

กราฟเทียบ 3 โมเดล: `results/model_comparison.png` (ซ้าย = accuracy ○ CV บน train ● test, กลาง = train time, ขวา = predict time) ตัวเลขดิบอยู่ใน `results/model_comparison.csv`

"Test Acc" มาจากโมเดลสุดท้ายตัวเดียว (Bagging ใช้ seed 0) ส่วน "Test Acc (10 seeds)" คือค่าเฉลี่ยของ 10 seed ที่ใช้จับเวลา

Confusion matrix บน test (แถว = จริง, คอลัมน์ = ทำนาย):
- Bagging, Boosting: `[[10,0,0],[0,9,1],[0,1,9]]` ผิด 2 ตัว (versicolor ↔ virginica)
- Decision Tree: `[[10,0,0],[0,9,1],[0,0,10]]` ผิด 1 ตัว

Decision Tree CV แยกตาม depth: 1 → 0.667, 2 → 0.942, **3 → 0.950**, 4/5/None → 0.933

### Sweep ของ ensemble (`results/ensemble_sweep.csv`)

Boosting ได้ลอง learning_rate 0.5 และ 1.0 ตามสเปก แต่ CV เลือก **lr=1.0** (CV 0.9667 ที่ n=10 สูงกว่าทุกค่าของ lr=0.5 ซึ่งสูงสุด 0.95) ตารางและกราฟจึงแสดงเฉพาะ lr=1.0 ผลของ lr=0.5 ยังอยู่ใน CSV

| Model | n_estimators | CV Acc | Test Acc (10 seeds) | Train ms | Predict ms |
|---|---|---|---|---|---|
| Bagging | 5 | **0.9583** | 0.9233 ± 0.0300 | 2.57 ± 0.14 | 0.092 ± 0.005 |
| Bagging | 10 | 0.9583 | 0.9267 ± 0.0291 | 5.19 ± 0.53 | 0.165 ± 0.010 |
| Bagging | 25 | 0.9500 | 0.9433 ± 0.0153 | 12.30 ± 0.58 | 0.393 ± 0.019 |
| Bagging | 50 | 0.9500 | 0.9333 ± 0.0149 | 24.61 ± 0.59 | 0.775 ± 0.048 |
| Bagging | 100 | 0.9500 | 0.9300 ± 0.0180 | 49.18 ± 1.65 | 1.487 ± 0.085 |
| Boosting lr=1.0 | 5 | 0.9500 | 0.9333 | 0.65 ± 0.06 | 0.029 ± 0.003 |
| Boosting lr=1.0 | 10 | **0.9667** | 0.9333 | 1.26 ± 0.07 | 0.050 ± 0.003 |
| Boosting lr=1.0 | 25 | 0.9333 | 0.9333 | 3.17 ± 0.13 | 0.111 ± 0.002 |
| Boosting lr=1.0 | 50 | 0.9333 | 0.9333 | 6.55 ± 0.42 | 0.216 ± 0.014 |
| Boosting lr=1.0 | 100 | 0.9250 | 0.9333 | 12.95 ± 0.68 | 0.421 ± 0.013 |

กราฟประกอบ (สเปกข้อ 8.7 ใช้ดูว่าขนาด ensemble มีผลอย่างไร และใช้เลือก n_estimators): `results/accuracy_vs_n_estimators.png` (ซ้าย = CV บน train, ขวา = test mean ± std 10 seeds; เส้นประสีเทา = Decision Tree เดี่ยว depth 3 ซึ่งไม่มี n_estimators จึงเป็นเส้นอ้างอิงแนวนอน) และ `results/time_vs_n_estimators.png`

### เทียบกับ sklearn (`results/sklearn_comparison.csv`, train data เดียวกัน, 10 seeds)

| Model | ของเรา test acc | sklearn test acc | ของเรา fit ms | sklearn fit ms |
|---|---|---|---|---|
| Decision Tree (เต็มความลึก) | 0.9333 | 0.9200 ± 0.0163 | 0.64 | 0.56 |
| Bagging n=50 | 0.9333 ± 0.0149 | 0.9333 | 24.3 | 44.7 |
| AdaBoost SAMME stump n=50 | 0.9333 | 0.9333 | 6.6 | 39.2 |

ต่างกันไม่เกิน 1.3 percentage point ผ่าน sanity check (ส่วน sklearn tree ที่มี std เป็นเพราะสุ่มลำดับ feature ตอนที่ split เสมอกัน)

## 4. วิเคราะห์

- **ทุกโมเดลอยู่ในช่วง 0.93–0.97 บน test** ซึ่ง test มีแค่ 30 ตัว ผิด 1 ตัว = 3.3 pp ความต่างที่เห็น (1 ตัวอย่าง) จึง **ไม่มีนัยสำคัญทางสถิติ** ตัวที่ผิดเป็นคู่ versicolor/virginica ที่ซ้อนกันอยู่ ส่วน setosa ถูกหมดทุกโมเดล
- **CV กับ test สลับอันดับกัน:** Boosting มี CV สูงสุด (0.967) แต่ Decision Tree depth 3 ได้ test สูงสุด (0.967) อย่างที่คาดไว้สำหรับชุดข้อมูลเล็ก ถ้าเลือกโมเดลจาก test จะเป็นการ overfit กับ test เราจึงยึดตาม CV
- **Bagging/Boosting ไม่แย่กว่า tree เดี่ยวอย่างมีนัยสำคัญ** ใน CV ทั้งคู่ดีกว่า tree (0.958/0.967 vs 0.950) แต่บน test ต่างกัน 1 ตัว Iris แยกง่ายจนตัว tree เองก็มี variance ต่ำอยู่แล้ว ensemble จึงไม่ค่อยมีอะไรให้ช่วย
- **Boosting เพิ่ม n แล้ว CV ไม่ดีขึ้น** (0.967 ที่ n=10 → 0.925 ที่ n=100) เพราะเริ่มไล่ fit จุดที่ซ้อนกันระหว่าง versicolor/virginica ขณะที่ test คงที่ 0.9333 ทุกค่า n
- **Bagging เพิ่ม n แล้ว variance ข้าม seed ลดลง** (std 0.030 → 0.018) แต่ค่าเฉลี่ยไม่ได้สูงขึ้น

### Trade-off เวลา vs accuracy

- เวลา fit/predict ของทั้งสอง ensemble **โตแบบเชิงเส้นตาม n_estimators** (เห็นได้จากกราฟ)
- **Bagging แพงกว่า Boosting ราว 4 เท่าที่ n เท่ากัน** เพราะแต่ละตัวเป็น tree เต็มความลึก (หลาย node) ส่วน Boosting ใช้แค่ stump (1 split) ถึง Boosting จะต้อง train ตามลำดับก็ยังเร็วกว่า
- tree เดี่ยวเร็วที่สุด (fit 0.41 ms, predict ~0.02 ms) และ accuracy ไม่ได้ด้อยกว่า ส่วน Boosting ที่เลือก (n=10) ช้ากว่า tree ราว 3 เท่า และ Bagging ที่เลือก (n=5) ช้ากว่าราว 6 เท่า
- **ข้อสรุป:** สำหรับ Iris tree เดี่ยว depth 3 ให้ accuracy เท่ากับ ensemble โดยใช้เวลาน้อยกว่าหลายเท่า ensemble จะคุ้มค่ากับชุดข้อมูลที่ใหญ่และมี noise มากกว่านี้ ถ้าต้องเลือก ensemble Boosting n≈10 ให้ CV ดีที่สุดในราคาต่ำสุด
- ค่าเวลาเป็นของเครื่องที่รัน ใช้เทียบกันในการรันเดียวกันได้ แต่อย่าเทียบข้ามเครื่อง

## 5. ข้อจำกัด

- test set เล็ก (30) ความไม่แน่นอนของ test accuracy ประมาณ ±4–5 pp
- ไม่ได้ทำ OOB score สำหรับ Bagging (เป็นทางเลือกในสเปก)
- ตัด SVM baseline ออกตามที่ผู้สั่งงานระบุว่างานต้องการเฉพาะ Decision Tree, Bagging และ Boosting
