# 🎯 Client Presentation Guide — Roman Urdu

> Yeh guide aap ke liye hai jab aap **client** ko yeh project present karein.
> Maqsad: client ko confidence dena ke kaam professional tareeqe se, schedule par,
> aur sahi direction mein ho raha hai. Technical jargan kam, **value aur progress** zyada.

**Project:** Classification of Malignant Melanoma in Canines (CATCH dataset)
**Phase abhi:** Week 1–2 (Foundation) — data ki tayyari mukammal aur tested.

---

## 🧭 PEHLE YEH SAMJHEIN — Client ko kya sunna hai?

Client 3 cheezein sunna chahta hai:
1. **Kaam ho raha hai?** → Haan, aur live chala kar dikhaunga.
2. **Schedule par hai?** → Haan, 12 hafton ke plan mein hum sahi jagah par hain.
3. **Aage kya?** → Clear roadmap hai (model training Week 3 se).

Baaqi sab technical tafseel sirf tab batayein jab client khud puche.

---

## 🎤 SEGMENT 1 — Opening (1 min)

> "Assalam-o-Alaikum. Aaj main aap ko is project ka pehla milestone dikhaunga.
> Project ka maqsad hai — kutton ki skin cancer (melanoma) ki microscope slides ko
> Artificial Intelligence se automatically pehchanna aur classify karna.
>
> Yeh sirf kutton ke liye nahi — kutton ka melanoma insaanon ke melanoma se bohat
> milta-julta hai, isliye yeh research medical field ke liye bhi qeemti hai.
>
> Poora project 12 hafton ka hai. Aaj main pehle do hafton ka mukammal kaam dikhaunga
> — jo hai data ki tayyari. Yeh sab se zaroori bunyad hai, kyunke achi tayyari ke
> baghair AI model achi tarah kaam nahi karta."

---

## 🎤 SEGMENT 2 — Hum kahan hain? (45 sec)

🖥️ **DIKHAO:** Slide deck ka "Where we are" wala page (Supervisor_Meeting PDF, slide 2).

> "Yeh project ka roadmap hai. Char phases hain:
> - **Foundation** (hafta 1–3) — data tayyar karna ← **abhi hum yahan hain**
> - **Model Development** (hafta 3–8) — AI model banana
> - **Evaluation** (hafta 8–10) — model ka test karna
> - **Documentation** (hafta 8–12) — final report
>
> Hum bilkul **schedule par** hain. Foundation phase ka data wala hissa tayyar aur
> tested hai."

---

## 🎤 SEGMENT 3 — Hum ne kya banaya? (2 min) ⭐

> "Hum ne ek **automatic data preparation pipeline** banaya hai. Iska matlab —
> ek factory line jaisa system jo raw microscope slides ko le kar, unhe AI ke liye
> tayyar dataset mein badal deta hai. Yeh char kaam khud-ba-khud karta hai:
>
> 1. **Quality Check** — har slide ko check karta hai: kahin blurry to nahi? kahin
>    khaali to nahi? Jo slide kharab ho, automatically alag kar deta hai. Isse
>    sirf achi quality ka data aage jaata hai.
>
> 2. **Color Standardisation (Macenko method)** — har lab aur machine ka color thoda
>    alag hota hai. Yeh step har slide ko ek standard color par le aata hai, taake AI
>    color ki wajah se confuse na ho. Yeh 2009 ki ek mash'hoor research par based hai.
>
> 3. **Patch Extraction** — ek slide bohat bari hoti hai, AI mein poori nahi aa sakti.
>    Isliye system use chote tukdon mein kaat deta hai, aur sirf woh tukde rakhta hai
>    jinme asal tissue ho — khaali background phenk diya jaata hai.
>
> 4. **Data Splitting** — taiyaar data ko teen hisson mein baant deta hai: 70% se AI
>    seekhega, 15% se hum check karenge, 15% final test ke liye. Yeh standard AI
>    practice hai.
>
> Khaas baat: yeh sab **automatic aur reproducible** hai — ek command se poora chal
> jaata hai, aur har baar same result deta hai."

---

## 🎤 SEGMENT 4 — LIVE Demo (2 min) ⭐⭐ SAB SE AHEM

> "Ab main aap ko yeh live chala kar dikhata hoon."

🖥️ **Terminal mein (pehle se test kar lena!):**
```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_week1_2_pipeline.py --demo
```

> "Bas ek command. Dekhiye — system ne 2 slides process kiye, dono quality check pass
> kiye, 26 tissue patches banaye, aur data ko 18 training / 4 validation / 4 test mein
> baant diya. Sirf chand second mein. Aur DONE aa gaya."

🟡 **Important — honesty wala point (client trust banta hai):**
> "Ek baat saaf kar doon: abhi yeh demo data par chala — yani test ke liye banawati
> slides. Asal CATCH dataset (750 real slides) TCIA se download hona baaqi hai, kyunke
> uske liye official agreement aur bara download chahiye. **Lekin system bilkul ready
> hai** — jaise hi real data aata hai, yehi system bina kisi tabdeeli ke real slides
> par chal jayega. Yani mehnat ka kaam ho chuka hai, sirf data plug karna hai."

---

## 🎤 SEGMENT 5 — Results dikhayein (1.5 min)

🖥️ **DIKHAO:** `outputs/figures/stain_normalization_demo_slide_bluish.png`

> "Yeh dekhiye natija. Left taraf original slide hai — neela color liye hue. Right
> taraf system ka normalised version — color standard medical pink-purple ho gaya.
> Yehi consistency AI ki accuracy barhati hai."

🖥️ **DIKHAO:** `outputs/figures/patches_demo_slide_pinkish.png`

> "Yeh tissue patches hain — slide ke saaf tukde, background nikaal kar."

🖥️ **DIKHAO:** `outputs/logs/quality_report.csv`

> "Aur yeh quality report hai — har slide ke objective numbers, taake faisla
> documented aur professional ho, sirf andaza nahi."

---

## 🎤 SEGMENT 6 — Quality aur Professionalism (1 min)

> "Main yeh bhi batana chahunga ke kaam professional standards par hua hai:
> - Saara code **modular aur organized** hai — aage barhana asaan hoga.
> - **Reproducible** hai — koi bhi banda exact yeh environment dobara bana sakta hai.
> - **Configurable** hai — settings ek jagah hain, code chhede baghair badli ja sakti
>   hain.
> - **Version controlled** (Git) — har tabdeeli ka record hai.
> - Aur poora kaam **documented** hai — report aur slides automatically generate hote
>   hain."

---

## 🎤 SEGMENT 7 — Aage ka plan (45 sec)

> "Agle do hafton mein (Week 3–5):
> 1. Real CATCH data download kar ke yehi pipeline us par chalana.
> 2. Phir **AI model** banana shuru karna — U-Net jo tumour ka exact area nikalta hai,
>    aur ResNet/EfficientNet jo tumour ki qism pehchanta hai.
>
> Yani ab data tayyar hai, agla qadam asal AI hai. Project mukammal schedule par hai."

---

## 🎤 SEGMENT 8 — Closing (30 sec)

> "Khulasa: hum ne is project ki mazboot bunyad rakh di hai — ek tested, automatic
> data preparation system jo real data ke liye ready hai. Hum schedule par hain aur
> agla phase AI model development hai.
>
> Aap ka koi sawal ho to main hazir hoon. Shukriya."

---

## 🛡️ MUSHKIL SAWALON KE JAWAB (client puch sakta hai)

**Q: "Abhi tak AI model kyun nahi bana?"**
> "AI model ki accuracy poori tarah data ki quality par depend karti hai. Industry
> best-practice yehi hai ke pehle data ko theek se tayyar kiya jaye. Agar yeh bunyad
> kamzor ho to model kabhi achi tarah kaam nahi karega. Hum ne pehle 2 hafte isi
> bunyad ko mazboot banaya — yeh waqt ki bachat hai, zaya nahi."

**Q: "Real data abhi tak kyun nahi?"**
> "CATCH dataset ek official medical archive (TCIA) par hai jiske liye data-use
> agreement aur bara download chahiye — yeh process chal raha hai. Is dauraan hum ne
> waqt zaya nahi kiya: poora system bana kar test kar liya, taake data aate hi foran
> aage barh sakein."

**Q: "Kya yeh real slides par kaam karega?"**
> "Ji haan. System isi tarah design kiya gaya hai ke real medical slides (.svs format)
> ko OpenSlide library se parhe. Demo sirf isliye use ki taake aap ko aaj chalti hui
> cheez dikha sakoon."

**Q: "Iska faida (business/medical value) kya hai?"**
> "Manual tor par pathologist ko har slide dekhne mein waqt lagta hai. Yeh system aage
> chal kar tumour ko automatically detect aur classify karega — tez, consistent, aur
> bina thakaawat. Aur kutton ka yeh research insaani melanoma ke liye bhi madadgaar
> hai."

**Q: "Accuracy kitni hogi?"**
> "Yeh hum Week 8–10 mein evaluation phase mein measure karenge — proper metrics
> (Dice score, accuracy, sensitivity) se. Abhi hum foundation par hain, isliye
> accuracy ka number dena jaldbazi hogi. Lekin literature ke mutabiq aise models 85%+
> tak ja sakte hain."

---

## ✅ PRESENTATION SE PEHLE CHECKLIST

1. Terminal pehle se ek baar test karein: `python scripts\run_week1_2_pipeline.py --demo`
   — taake demo mein error na aaye.
2. `outputs/figures/` ki images pehle se khol kar rakhein (ya jaldi khol sakein).
3. Slide deck PDF (`docs/slides/Supervisor_Meeting_Week1-2.pdf`) khula rakhein.
4. VS Code font bara karein (`Ctrl + +`) taake screen share par saaf nazar aaye.
5. Internet/laptop charge check karein agar online meeting hai.
6. Aaram se, confidence se bolein — kaam achi tarah hua hai, bas usay clearly batana hai.

---

## 🔑 EK LINE MEIN (agar client jaldi mein ho)

> "Hum ne kutton ke cancer detection AI project ki bunyad mukammal kar li hai —
> ek automatic system jo medical slides ki quality check karta hai, unka color
> standard karta hai, unhe AI ke liye tayyar karta hai, aur train/test mein baant
> deta hai. Sab tested aur schedule par. Agla qadam: AI model banana."
