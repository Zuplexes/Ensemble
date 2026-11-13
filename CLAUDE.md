# CLAUDE.md — โปรเจกต์ Decision Tree / Bagging / Boosting บน Iris Dataset

ไฟล์นี้คือคำสั่งงานสำหรับ Claude Code อ่านทั้งหมดก่อนเริ่มลงมือ และทำตามลำดับ Phase ที่ระบุ

## 1. เป้าหมาย

1. ทำ **Decision Tree เดี่ยว** (เขียนเอง) เป็น baseline บน Iris
2. เขียน **Bagging เอง** (อัลกอริทึมหลักเขียนด้วย numpy + built-in เท่านั้น) และ **วัดเวลา** train/predict
3. เขียน **Boosting เอง** (AdaBoost แบบ multi-class SAMME, อัลกอริทึมหลักเขียนด้วย numpy + built-in เท่านั้น)
4. เปรียบเทียบทุกโมเดลในตารางเดียว (accuracy + เวลา) พร้อมกราฟและรายงานสรุป
5. สร้างไฟล์ `requirements.txt` ให้โปรเจกต์รันซ้ำได้

> **ไม่มี SVM:** ผู้ใช้ยืนยัน (2026-11-13) ว่างานต้องการเฉพาะ Decision Tree, Bagging และ Boosting จึงตัด SVM baseline (`svm_experiment.py`) ออกแล้ว ห้ามเพิ่มกลับเว้นแต่ผู้ใช้สั่ง

## 2. Dataset

- แหล่งที่มา: https://huggingface.co/datasets/scikit-learn/iris
- ดาวน์โหลดไฟล์ CSV (`Iris.csv`) ไปไว้ที่ `data/Iris.csv` (ถ้าลิงก์ตรงใช้ไม่ได้ ให้เปิดหน้า dataset แล้วหา path ของไฟล์ CSV จริง หรือถามผู้ใช้ให้วางไฟล์เอง)
- 150 แถว, 4 features (SepalLength, SepalWidth, PetalLength, PetalWidth), 3 classes (setosa, versicolor, virginica)
- มีคอลัมน์ `Id` ให้ตัดทิ้ง และแปลงชื่อคลาสเป็น 0, 1, 2

## 3. กฎการใช้ Library

**อนุญาตให้ใช้ `scikit-learn` ได้** แต่มีข้อจำกัดเฉพาะส่วนที่โจทย์สั่งให้เขียนเอง

### ส่วนที่ต้องเขียนเอง (numpy + built-in เท่านั้น)
- `bagging.py` และ `boosting.py` (ตัวอัลกอริทึม Bagging และ AdaBoost/SAMME)
- `tree.py` (Decision Tree / Decision Stump ที่ใช้เป็น base learner รองรับ `sample_weight`) เขียนเองด้วย เพื่อให้ ensemble ทั้งชุดไม่พึ่ง library
- ในสามไฟล์นี้ใช้ได้เฉพาะ **`numpy`** และ **standard library ของ Python** (`time`, `math`, `random`, `collections`, ฯลฯ)
- **ห้าม** import `sklearn`, `scipy`, `pandas` หรือ ML library ใดๆ ในสามไฟล์นี้

### ส่วนที่ใช้ scikit-learn ได้
- `data_utils.py`: `train_test_split` (stratified), `StandardScaler`, `StratifiedKFold` ใช้ของ sklearn ได้ (อ่านไฟล์ CSV ด้วย `csv` module หรือ pandas ก็ได้)
- `metrics.py`: `accuracy_score`, `confusion_matrix`, `classification_report` ฯลฯ
- `compare_sklearn.py`: เทียบ `BaggingClassifier` / `AdaBoostClassifier` / `DecisionTreeClassifier` ของ sklearn กับของที่เขียนเอง เพื่อ sanity check
- `main.py`: ใช้ `matplotlib` วาดกราฟ (โค้ดวาดกราฟอยู่ใน `main.py` ไม่แยกไฟล์) และ `pandas` จัดตารางผลได้

> ถ้าผู้ใช้สั่งภายหลังว่าอนุญาตให้ใช้ sklearn ในส่วน Bagging/Boosting ด้วย ให้ปรับตามที่สั่ง แต่ค่าเริ่มต้นคือเขียนเอง

## 4. การแบ่งข้อมูล Train/Test 80/20 (สำคัญ)

ทำใน `data_utils.py` โดยใช้ `sklearn.model_selection.train_test_split` ได้ ตามข้อกำหนดนี้

- สัดส่วน **train 80% / test 20%** (`test_size=0.2`) จาก 150 แถว ได้ **train = 120, test = 30**
- ต้องเป็นแบบ **stratified** (`stratify=y`) ทุกคลาสถูกแบ่ง 80/20 เท่ากัน (train 40 ต่อคลาส, test 10 ต่อคลาส)
- ใช้ **random seed คงที่ = 42** (`random_state=42`) เพื่อให้ผลทำซ้ำได้
- แบ่งเพียง **ครั้งเดียว** แล้วทุกโมเดล (Decision Tree, Bagging, Boosting) ต้องใช้ชุดเดียวกัน เพื่อให้เทียบกันได้ยุติธรรม
- บันทึกดัชนีหรือไฟล์ split ไว้ (เช่น `data/split_indices.npz`) เพื่อใช้ซ้ำ
- **Standardization**: fit จาก **train เท่านั้น** แล้ว transform ทั้ง train และ test (กัน data leakage)
- **Test set ห้ามแตะ** ระหว่างการเลือก hyperparameter ใช้ **stratified 5-fold CV บน train set เท่านั้น** แล้วรายงานผล test ครั้งสุดท้ายเพียงครั้งเดียวต่อโมเดล
- ต้องมี test/assertion ตรวจ: ขนาด 120/30, ไม่มีแถวซ้ำข้าม train/test, สัดส่วนคลาสถูกต้อง

## 5. requirements.txt และการตั้งค่าสภาพแวดล้อม

ต้องสร้างไฟล์ `requirements.txt` ที่ root ของโปรเจกต์ (ทำใน Phase 1) โดย

- ใส่เฉพาะ package ที่โปรเจกต์ใช้จริง ได้แก่ `numpy`, `scikit-learn`, `matplotlib`, `pandas` (ถ้าใช้), `pytest`
- ระบุเวอร์ชันแบบ pin (`==`) ตามที่ติดตั้งและทดสอบจริงในเครื่อง (ตรวจด้วย `pip freeze` หรือ `pip show` แล้วคัดมาเฉพาะ package ที่ใช้) ห้ามเดาเวอร์ชันเอง
- สร้าง virtual environment (`python -m venv .venv`) แล้วติดตั้งด้วย `pip install -r requirements.txt` เพื่อทดสอบว่ารันได้จริง
- ถ้าเพิ่มหรือเลิกใช้ library ภายหลัง ต้องอัปเดต `requirements.txt` ให้ตรงเสมอ
- เขียนวิธีติดตั้งและรันสั้นๆ ไว้ใน `REPORT.md` หรือ `README.md`

ตัวอย่างรูปแบบ (เวอร์ชันให้ใช้ตามที่ติดตั้งจริง ไม่ใช่ตามตัวอย่างนี้):

```
numpy==<installed-version>
scikit-learn==<installed-version>
matplotlib==<installed-version>
pandas==<installed-version>
pytest==<installed-version>
```

## 6. โครงสร้างไฟล์

```
project/
├── CLAUDE.md
├── requirements.txt
├── data/
│   ├── Iris.csv
│   └── split_indices.npz
├── data_utils.py        # load CSV, encode label, stratified split 80/20, scaler, k-fold
├── tree.py              # DecisionTree (CART, Gini) + DecisionStump, รองรับ sample_weight (เขียนเอง)
├── bagging.py           # Bagging เขียนเอง
├── boosting.py          # AdaBoost (SAMME) เขียนเอง
├── metrics.py           # accuracy, confusion matrix, precision/recall/F1
├── compare_sklearn.py   # เทียบกับ sklearn (sanity check)
├── tests/               # unit tests
├── main.py              # รันทดลองทั้งหมด + จับเวลา + สร้างตาราง/กราฟ
├── results/             # ตาราง CSV/JSON และกราฟ PNG
└── REPORT.md            # รายงานสรุป
```

## 7. สเปกของแต่ละโมดูล

### 7.1 `data_utils.py`
- `load_iris(path)` อ่านไฟล์ คืน `X` (float64, shape 150×4), `y` (int, 0/1/2)
- `get_split(seed=42)` คืน `X_train, X_test, y_train, y_test` ตามข้อ 4 (แบ่งครั้งเดียวและโหลดซ้ำจากไฟล์ที่บันทึกไว้)
- ฟังก์ชัน scale (fit บน train เท่านั้น) และสร้าง stratified k-fold indices

### 7.2 `tree.py` (เขียนเอง)
- `DecisionTree(max_depth=None, min_samples_split=2)` และ `DecisionStump` (depth = 1)
- เกณฑ์แบ่ง: Gini impurity (ถ่วงน้ำหนักด้วย `sample_weight`)
- API: `fit(X, y, sample_weight=None)`, `predict(X)`
- หา threshold แบบ vectorized (sort แล้ว cumulative sum) เพื่อความเร็ว
- ต้อง deterministic เมื่อ seed เดียวกัน

### 7.3 `bagging.py` (เขียนเอง)
- `Bagging(base_estimator_factory, n_estimators=10, seed=None)`
- วนลูปสุ่ม bootstrap sample (แบบคืนที่ ขนาดเท่า train) ด้วย `numpy.random.default_rng(seed)`
- train base learner แต่ละตัวบน bootstrap ของตัวเอง
- predict ด้วย majority vote (`np.bincount` + `argmax`)
- (ทางเลือก) คำนวณ OOB score
- **วัดเวลา** ด้วย `time.perf_counter()` แยก `fit_time` และ `predict_time`

### 7.4 `boosting.py` (เขียนเอง)
- AdaBoost แบบ multi-class **SAMME**
- เริ่ม weight = 1/N, วนรอบละ 1 base learner (stump หรือ tree ตื้น)
- `alpha = learning_rate * (ln((1 - err) / err) + ln(K - 1))`
- อัปเดต weight ของตัวอย่างที่ทายผิดแล้ว normalize ให้รวม = 1
- จัดการกรณี `err == 0` (หยุดหรือใช้ epsilon) และ `err >= 1 - 1/K` (หยุดหรือข้ามรอบ)
- predict แบบ weighted vote ด้วย alpha
- วัดเวลา fit/predict เช่นเดียวกับ Bagging

## 8. การทดลองและการวัดเวลา

1. **Baseline:** Decision Tree เดี่ยว (ของที่เขียนเอง)
2. **Bagging:** แปรค่า `n_estimators` = 5, 10, 25, 50, 100
3. **Boosting:** แปรค่า `n_estimators` = 5, 10, 25, 50, 100 และ `learning_rate` = 0.5, 1.0
4. **เวลา:** รันซ้ำ **10 ครั้ง** (ต่าง seed) แล้วรายงาน **mean ± std** ของ train time และ predict time (ใช้ `time.perf_counter()`)
5. **Metric:** accuracy, macro precision/recall/F1, confusion matrix บน test set (ใช้ hyperparameter ที่เลือกจาก CV บน train)
6. **ตารางสรุปสุดท้าย:**

| Model | CV Acc (train) | Test Acc | Train time (mean±std) | Predict time (mean±std) |
|---|---|---|---|---|
| Decision Tree เดี่ยว | | | | |
| Bagging (เขียนเอง) | | | | |
| Boosting (เขียนเอง) | | | | |

7. **กราฟ:**
   - **กราฟหลัก** `results/model_comparison.png`: เทียบโมเดลสุดท้ายทั้ง 3 ตัวโดยตรง แบ่งเป็นแผง accuracy (CV บน train และ test), train time และ predict time แยกกัน (ไม่ใช้แกนคู่)
   - **กราฟประกอบ:** accuracy vs n_estimators (แยกแผง CV บน train กับ test) และเวลา vs n_estimators สำหรับ Bagging และ Boosting โดย Boosting แสดงเฉพาะ learning_rate ที่ CV เลือก (lr อื่นยังเก็บใน CSV) และ Decision Tree เดี่ยวเป็นเส้นประแนวนอนไว้อ้างอิง
8. บันทึกผลลง `results/` เป็น CSV/JSON/PNG

## 9. การตรวจสอบความถูกต้อง (ต้องผ่านก่อนไป Phase ถัดไป)

- unit test tree บนข้อมูลเล็กๆ ที่รู้คำตอบ
- unit test: weight รวม = 1 หลังอัปเดตแต่ละรอบใน AdaBoost
- unit test: split 80/20 ถูกต้องตามข้อ 4
- ผลที่เขียนเองควรใกล้เคียง sklearn (ต่างกันไม่เกินไม่กี่เปอร์เซ็นต์) และ Bagging/Boosting ไม่ควรแย่กว่า tree เดี่ยวอย่างมีนัยสำคัญ
- ใช้ `pytest` รันเทสต์
- ตรวจด้วย `grep` ว่าไม่มี import `sklearn` / `scipy` / `pandas` ใน `tree.py`, `bagging.py`, `boosting.py`
- ตรวจว่า `pip install -r requirements.txt` ใน venv ใหม่แล้ว `python main.py` และ `pytest` รันผ่าน

## 10. ลำดับการทำงาน (Phases)

ทำทีละ Phase จบแล้วสรุปสั้นๆ และรันเทสต์ก่อนไปต่อ

1. **Phase 1:** เตรียมโปรเจกต์ + venv + **`requirements.txt`** + ดาวน์โหลดข้อมูล + `data_utils.py` (load, split 80/20, scaler, k-fold) + เทสต์
2. **Phase 2:** `metrics.py` (evaluate, CV accuracy, จับเวลา)
3. **Phase 3:** `tree.py` (Decision Tree / Stump + sample_weight) + เทสต์
4. **Phase 4:** `bagging.py` + จับเวลา
5. **Phase 5:** `boosting.py` (SAMME) + เทสต์
6. **Phase 6:** `main.py` รันทดลองทั้งหมด สร้างตาราง/กราฟ + `compare_sklearn.py`
7. **Phase 7:** อัปเดต `requirements.txt` ให้ตรงกับที่ใช้จริง + เขียน `REPORT.md` สรุปวิธีการ ผลลัพธ์ การวิเคราะห์ trade-off เวลา vs accuracy

## 11. แนวทางเขียนโค้ด

- Python 3.10+, มี type hints และ docstring สั้นๆ ใช้ภาษาให้สม่ำเสมอ
- ใช้ `np.random.default_rng(seed)` ทุกที่ที่ต้องสุ่ม ห้ามพึ่ง global random state
- ใช้ vectorization ของ numpy แทน loop ซ้อนเมื่อทำได้
- โค้ดอ่านง่าย ฟังก์ชันสั้น ไม่ over-engineer
- รัน `python main.py` ครั้งเดียวต้องได้ผลครบทุกอย่างและทำซ้ำได้

## 12. สิ่งที่ห้ามทำ

- ห้ามใช้ test set เลือก hyperparameter หรือปรับโมเดล
- ห้าม fit scaler บนข้อมูลทั้งหมดก่อน split
- ห้ามใช้ sklearn / scipy / pandas ใน `tree.py`, `bagging.py`, `boosting.py`
- ห้ามเปลี่ยนสัดส่วน split หรือ seed โดยไม่ถามผู้ใช้
- ห้ามข้าม Phase หรือรายงานผลโดยไม่รันจริง
- ห้ามลืมอัปเดต `requirements.txt` เมื่อมีการเพิ่ม library
- หากข้อกำหนดไม่ชัดเจนหรือขัดแย้งกัน ให้ถามผู้ใช้ก่อนตัดสินใจเอง

## 13. สิ่งที่ต้องส่งมอบ

- โค้ดครบตามโครงสร้างในข้อ 6
- `requirements.txt` ที่ติดตั้งและรันได้จริง
- เทสต์ผ่านทั้งหมด
- โฟลเดอร์ `results/` ที่มีตาราง กราฟ และเวลาที่วัดได้
- `REPORT.md` ที่อธิบายวิธีทำ ผลการทดลอง และข้อสรุป
