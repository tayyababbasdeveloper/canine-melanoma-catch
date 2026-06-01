# 🎬 Video Explanation Guide — Roman Urdu (Week 1–2)

Yeh script aap ke video ke liye hai. Har segment mein 2 cheezein hain:
- 🖥️ **DIKHAO** = screen par kya dikhana hai
- 🎤 **BOLO** = kya bolna hai (Roman Urdu)

Total video ~8–12 minute. Aaram se, ek-ek cheez.

---

## 🎬 SEGMENT 0 — Intro (30 sec)

🖥️ **DIKHAO:** Apna chehra / ya seedha VS Code khula hua.

🎤 **BOLO:**
> "Assalam-o-Alaikum. Main Muhammad Tayyab Abbas. Yeh mera MSc research project
> hai — University of Hull, module 771765. Project ka naam hai:
> *Classification of Malignant Melanoma in Canines* — yani kutton ki skin cancer
> ki slides ko deep learning se classify karna, CATCH dataset use karke.
> Aaj main aap ko Week 1 aur Week 2 ka kaam step by step dikhaunga."

---

## 🎬 SEGMENT 1 — Project kis baare mein hai (1 min)

🖥️ **DIKHAO:** `Final_Research_Proposal` PDF ya bas baat karein.

🎤 **BOLO:**
> "Sab se pehle yeh samajhna zaroori hai ke project kya hai. Kutton mein melanoma —
> khaas tor par non-UV melanoma — insaanon ke melanoma se bohat milta-julta hai.
> Isliye kutton ka data insaani research ke liye bhi faida-mand hai. Iss ko
> comparative oncology kehte hain.
>
> Hamare paas CATCH dataset hai — 750 whole-slide histopathology images, jo expert
> pathologists ne annotate ki hain. Mera plan hai do kaam karna:
> ek, **segmentation** — U-Net se tumour ka area nikalna. Do, **classification** —
> ResNet-50 aur EfficientNet-B3 se tumour ki qism pehchanna.
>
> Lekin yeh poora project 12 hafton ka hai. Aaj main sirf **pehle do hafton** ka
> kaam dikha raha hoon — jo hai Foundation phase, yani data ko taiyaar karna."

---

## 🎬 SEGMENT 2 — Hum 12 hafton mein kahan hain (45 sec)

🖥️ **DIKHAO:** Proposal ka Gantt chart (Figure 4).

🎤 **BOLO:**
> "Yeh raha project ka Gantt chart. Project 26 May se 14 August tak chalta hai.
> Char phases hain:
> Foundation hafta 1 se 3, Model Development 3 se 8, Evaluation 8 se 10, aur
> Documentation 8 se 12.
>
> Abhi hum Week 2 par hain — yani Foundation phase mein. Is phase mein **abhi koi
> AI model train nahi hota**. Pehle data ko saaf aur taiyaar kiya jaata hai, taake
> baad mein model achi tarah seekh sake. Aaj main wahi data preparation dikhaunga."

---

## 🎬 SEGMENT 3 — Folder structure (1.5 min)

🖥️ **DIKHAO:** VS Code ka left sidebar (file explorer) — ek-ek folder click karke kholo.

🎤 **BOLO:**
> "Yeh mera project ka folder structure hai. Main ne ise professional tarीqe se
> organize kiya hai. Aaiye ek-ek folder dekhte hain:
>
> - **`.venv`** — yeh virtual environment hai. Iski wajah se mere project ke saare
>   Python packages alag-thalag rehte hain, mere baaki projects par koi asar nahi
>   padta. Yeh bohat zaroori best practice hai.
>
> - **`config`** — isme `config.yaml` file hai. Yahan saari settings ek jagah hain —
>   jaise patch ka size, quality ke thresholds. Code change kiye bagair main yahan
>   se values badal sakta hoon.
>
> - **`data`** — isme `raw` folder asal slides ke liye hai, aur `processed` folder
>   mein taiyaar shuda patches aur splits aate hain.
>
> - **`src`** — yeh asal code hai. Teen hisse: `data_acquisition`, `preprocessing`,
>   aur `utils`.
>
> - **`scripts`** — yeh chalane wali files hain.
>
> - **`outputs`** — yahan saare results aate hain: `figures` mein images, `logs`
>   mein reports.
>
> - **`docs`** — yahan meri report, slides, aur literature notes hain."

---

## 🎬 SEGMENT 4 — Environment Setup (Week 1 ka kaam) (1 min)

🖥️ **DIKHAO:** `requirements.txt` aur `.venv` folder.

🎤 **BOLO:**
> "Week 1 ka pehla kaam tha environment setup. Main ne ek virtual environment banaya
> `python -m venv .venv` command se. Phir saari zaroori libraries install kiin —
> jaise NumPy, OpenCV, scikit-image, scikit-learn, aur OpenSlide jo medical slides
> parhne ke liye hai.
>
> Yeh saari libraries `requirements.txt` mein likhi hain, aur exact versions
> `requirements-lock.txt` mein. Iska faida yeh hai ke koi bhi banda meri yeh files
> le kar bilkul wohi environment dobara bana sakta hai — isay reproducibility kehte
> hain, jo research mein bohat important hai."

🖥️ **DIKHAO:** Terminal mein type karke dikhao:
```powershell
.\.venv\Scripts\Activate.ps1
```
🎤 **BOLO:**
> "Har baar kaam shuru karne se pehle main environment activate karta hoon. Dekhiye,
> ab prompt par `(.venv)` likha aa gaya — iska matlab environment active hai."

---

## 🎬 SEGMENT 5 — Literature Review (Week 1–2) (45 sec)

🖥️ **DIKHAO:** `docs/literature/literature_review_notes.md` kholo.

🎤 **BOLO:**
> "Doosra kaam tha literature review refine karna. Yahan main ne apne project se
> juRi tamaam important research papers ki summary likhi — har paper ka method aur
> uska natija, aur yeh ke woh mere kaam ko kaise justify karta hai.
>
> Misaal ke tor par: U-Net segmentation ke liye, ResNet aur EfficientNet
> classification ke liye, aur Macenko method color normalisation ke liye. Yeh saare
> mere methodology ki bunyad hain."

---

## 🎬 SEGMENT 6 — Preprocessing Code samjhana (3 min) ⭐ SAB SE AHEM

🖥️ **DIKHAO:** `src/preprocessing/` folder kholo, har file baari baari.

🎤 **BOLO (intro):**
> "Ab aata hai sab se important hissa — data preprocessing ka code. Slide se le kar
> taiyaar dataset tak char steps hain. Main har file dikhata hoon."

### 6.1 — Quality Check
🖥️ **DIKHAO:** `src/data_acquisition/quality_check.py`
🎤 **BOLO:**
> "Pehla step: quality check. Yeh file har slide ko check karti hai — kya woh blurry
> hai? kya isme tissue kam hai? kya yeh bohat dark ya bohat safed hai? Blur measure
> karne ke liye main *variance of Laplacian* use karta hoon. Jo slide kharab ho, use
> automatically flag karke alag kar diya jaata hai, ek waja ke saath."

### 6.2 — Stain Normalisation (Macenko)
🖥️ **DIKHAO:** `src/preprocessing/stain_normalization.py`
🎤 **BOLO:**
> "Doosra step — aur sab se ahem — stain normalisation. Masla yeh hai ke har lab aur
> scanner ka color thoda alag hota hai. Agar model ko alag-alag colors milein to woh
> confuse ho jaata hai. Macenko method, jo 2009 ki ek famous research hai, har slide
> ko ek standard H&E color appearance par le aata hai.
>
> Yeh kaam optical density space mein hota hai — pehle color ko OD mein convert
> karte hain, phir do main stain vectors (haematoxylin aur eosin) estimate karte
> hain, aur phir ek fixed reference par dobara map karte hain. Thori der mein main
> iska result bhi dikhaunga — before aur after."

### 6.3 — Patch Extraction
🖥️ **DIKHAO:** `src/preprocessing/patch_extraction.py` + `tissue.py`
🎤 **BOLO:**
> "Teesra step: patch extraction. Whole-slide image itni bari hoti hai ke poori
> model mein nahi aa sakti. Isliye main ise 256 by 256 ke chote tukdon mein kaat
> deta hoon. Lekin sirf woh tukde rakhta hoon jisme kam az kam 50% tissue ho —
> khaali safed background wale tukde phenk diye jaate hain. Tissue detect karne ke
> liye main saturation channel aur Otsu thresholding use karta hoon."

### 6.4 — Dataset Split
🖥️ **DIKHAO:** `src/preprocessing/dataset_split.py`
🎤 **BOLO:**
> "Chautha step: dataset ko teen hisson mein baatna — 70% training, 15% validation,
> 15% testing. Main *stratified* split use karta hoon, iska matlab har class ka
> proportion teeno hisson mein barabar rehta hai. Result train, val, aur test CSV
> files mein save hota hai."

---

## 🎬 SEGMENT 7 — Pipeline LIVE chalana (2 min) ⭐ DEMO

🖥️ **DIKHAO:** Terminal. Type karo aur run karo:
```powershell
python scripts\run_week1_2_pipeline.py --demo
```
🎤 **BOLO (jab chal raha ho):**
> "Ab main poori pipeline live chala raha hoon. Yeh `run_week1_2_pipeline` script
> upar ke chaaron steps ko ek saath jod deti hai.
>
> `--demo` flag ka matlab — abhi real CATCH data download nahi hua, isliye yeh
> synthetic, yani banawati, H&E slides khud bana kar test karta hai. Jaise hi real
> data aa jayega, main `--demo` ki jagah `--input data\raw` likh kar exact yehi
> pipeline real slides par chala sakta hoon, bina code change kiye.
>
> Dekhiye output — do slides process huin, dono QA pass huin, 26 patches bane, aur
> split hua 18 training, 4 validation, 4 test. Aur DONE aa gaya. Pipeline kaamyab."

🎤 **(Agar yaad ho to yeh bhi bata dein — honesty ke liye):**
> "Aik baat clear kar doon — yeh demo data banawati hai, sirf code test karne ke
> liye. Asal CATCH data TCIA se download hona baaki hai. Lekin code bilkul ready
> hai."

---

## 🎬 SEGMENT 8 — Results dikhana (1.5 min)

🖥️ **DIKHAO:** `outputs/figures/stain_normalization_demo_slide_bluish.png` kholo.
🎤 **BOLO:**
> "Yeh raha sab se ahem result — stain normalisation ka before aur after. Left taraf
> original slide hai jiska color neela-pan liye hue hai. Right taraf normalised
> version hai — dekhiye color standard H&E pink-purple ho gaya. Yeh exactly woh
> consistency hai jo model ki training ke liye chahiye."

🖥️ **DIKHAO:** `outputs/figures/patches_demo_slide_pinkish.png`
🎤 **BOLO:**
> "Yeh extracted patches hain — slide ke chote 256 by 256 ke tukde. Background wale
> khaali tukde khud-ba-khud nikaal diye gaye."

🖥️ **DIKHAO:** `outputs/logs/quality_report.csv` aur `split_summary.json`
🎤 **BOLO:**
> "Aur yeh reports hain — quality report CSV mein har slide ke numbers hain: width,
> height, brightness, blur score, tissue fraction, aur pass hua ya flag. Aur split
> summary JSON mein train, val, test ki ginti hai."

---

## 🎬 SEGMENT 9 — Report (DOCX) aur Slides (PDF) (1.5 min)

🖥️ **DIKHAO:** `docs/progress_report/Progress_Report_Week1-2.docx` (Word mein kholo).
🎤 **BOLO:**
> "Yeh mera progress report hai, Word document. Isme planned vs actual ka table hai,
> har kaam ki tafseel, artefacts ki list, risks aur unki mitigation, agle hafton ka
> plan, aur supervisor ke liye sawalat. Yeh report main ne `make_deliverables.py`
> script se automatically generate ki — python-docx library use karke."

🖥️ **DIKHAO:** `docs/slides/Supervisor_Meeting_Week1-2.pdf` kholo, slides scroll karo.
🎤 **BOLO:**
> "Aur yeh supervisor meeting ke liye slide deck hai, PDF format mein — 12 slides.
> Isme bhi sab kuch hai: hum Gantt mein kahan hain, environment, literature,
> pipeline, aur natije. Dekhiye slide 7 par before-after image embedded hai, aur
> slide 8 par patches. Yeh bhi usi script se automatically bani — reportlab library
> use karke."

🖥️ **DIKHAO:** Terminal:
```powershell
python scripts\make_deliverables.py
```
🎤 **BOLO:**
> "Yeh ek command se dono cheezein — report aur slides — dobara ban jaati hain. Sab
> automatic hai."

---

## 🎬 SEGMENT 10 — Khatma / Next steps (45 sec)

🎤 **BOLO:**
> "To yeh tha mera Week 1 aur Week 2 ka kaam: environment setup, literature review,
> aur ek poori working data preprocessing pipeline — jo quality check, stain
> normalisation, patch extraction, aur dataset split karti hai. Sab kuch test ho
> chuka hai aur real data ke liye ready hai.
>
> Agla kaam Week 3 se shuru hoga — U-Net segmentation model banana. Shukriya!"

---

## 📝 BONUS — Recording ke liye chote tips

1. **Screen recorder:** Windows mein `Win + G` (Xbox Game Bar) ya OBS Studio (free).
2. Video se pehle terminal **clear** kar lein: `cls` likh kar Enter.
3. Bolne se pehle ek baar yeh script parh lein taake flow smooth rahe.
4. Pipeline command pehle se ek baar test kar lein taake video mein error na aaye.
5. Font bara kar lein (VS Code: `Ctrl + +`) taake video mein code saaf nazar aaye.
6. Aaram se bolein — ek-ek cheez. Agar atak jayein to dobara record kar lein.

---

## 🔑 Ek line mein poora project (agar koi puche)

> "Main ne kutton ki cancer histopathology slides ke liye ek data preprocessing
> pipeline banai — jo slides ki quality check karti hai, unka color standard karti
> hai (Macenko), unhe chote patches mein kaatti hai, aur train/val/test mein baant
> deti hai — taake aage AI model train kiya ja sake."
