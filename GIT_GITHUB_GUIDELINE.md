# Git & GitHub Guideline
## คู่มือการใช้ Git และ GitHub สำหรับโปรเจค WangChang Model Classification

### 📋 สารบัญ
1. [การเตรียมความพร้อม](#การเตรียมความพร้อม)
2. [การ Setup Git Repository](#การ-setup-git-repository)
3. [การจัดการไฟล์ที่ไม่ต้องการ Push](#การจัดการไฟล์ที่ไม่ต้องการ-push)
4. [ขั้นตอนการ Push ไฟล์ครั้งแรก](#ขั้นตอนการ-push-ไฟล์ครั้งแรก)
5. [ขั้นตอนการ Push ไฟล์ในครั้งถัดไป](#ขั้นตอนการ-push-ไฟล์ในครั้งถัดไป)
6. [การจัดการ Branch](#การจัดการ-branch)
7. [คำสั่ง Git ที่ควรรู้](#คำสั่ง-git-ที่ควรรู้)
8. [Best Practices](#best-practices)
9. [การแก้ปัญหาที่พบบ่อย](#การแก้ปัญหาที่พบบ่อย)

---

## การเตรียมความพร้อม

### 1. ติดตั้ง Git
```bash
# ตรวจสอบว่าติดตั้ง Git แล้วหรือยัง
git --version

# หากยังไม่มี ให้ดาวน์โหลดจาก https://git-scm.com/
```

### 2. ตั้งค่า Git Configuration
```bash
# ตั้งชื่อผู้ใช้
git config --global user.name "ชื่อของคุณ"

# ตั้งอีเมล
git config --global user.email "your.email@example.com"

# ตรวจสอบการตั้งค่า
git config --list
```

### 3. สร้าง Repository บน GitHub
1. เข้าไปที่ GitHub.com
2. คลิก "New repository"
3. ตั้งชื่อ repository เช่น `wangchang-news-classification`
4. เลือก Public หรือ Private
5. **อย่า** เลือก "Initialize with README" หากมีไฟล์อยู่แล้ว
6. คลิก "Create repository"

---

## การ Setup Git Repository

### ขั้นตอนที่ 1: Initialize Git Repository
```bash
# เข้าไปในโฟลเดอร์โปรเจค
cd "d:\Python\HandOn_LLMs\wangchang_model_classify_new"

# สร้าง git repository
git init
```

### ขั้นตอนที่ 2: เชื่อมต่อกับ GitHub Repository
```bash
# เชื่อมต่อกับ remote repository
git remote add origin https://github.com/USERNAME/REPOSITORY_NAME.git

# ตรวจสอบ remote
git remote -v
```

---

## การจัดการไฟล์ที่ไม่ต้องการ Push

### สร้างไฟล์ .gitignore
```bash
# สร้างไฟล์ .gitignore
touch .gitignore
```

### เนื้อหาในไฟล์ .gitignore (แนะนำสำหรับโปรเจคนี้)
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Model Files (ขนาดใหญ่)
*.safetensors
*.pt
*.pth
*.bin
*.h5
*.ckpt

# Data Files (หากไม่ต้องการ share)
*.csv
!requirements.txt

# Model Directories (ขนาดใหญ่)
wcberta-prachathai67k/
wcberta-prachathai67k-best/
checkpoints/
models/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Logs
*.log
logs/
```

⚠️ **หมายเหตุสำคัญ**: Model files และ checkpoint files มักมีขนาดใหญ่มาก GitHub มีขีดจำกัดที่ 100MB ต่อไฟล์

---

## ขั้นตอนการ Push ไฟล์ครั้งแรก

### ขั้นตอนที่ 1: เพิ่มไฟล์ที่ต้องการ
```bash
# ดูสถานะไฟล์
git status

# เพิ่มไฟล์ทั้งหมด (ยกเว้นที่อยู่ใน .gitignore)
git add .

# หรือเพิ่มไฟล์เฉพาะ
git add app.py requirements.txt README.md

# ตรวจสอบไฟล์ที่เพิ่มแล้ว
git status
```

### ขั้นตอนที่ 2: Commit การเปลี่ยนแปลง
```bash
# Commit พร้อมข้อความ
git commit -m "Initial commit: Add news classification model project"

# หรือเขียนข้อความยาวๆ
git commit -m "feat: Add Thai news classification model

- Add fine-tuned WangChang BERT model
- Add training and validation scripts
- Add data preprocessing utilities
- Add requirements file"
```

### ขั้นตอนที่ 3: Push ขึ้น GitHub
```bash
# Push ครั้งแรก
git push -u origin main

# หรือหาก branch หลักชื่อ master
git push -u origin master
```

---

## ขั้นตอนการ Push ไฟล์ในครั้งถัดไป

### Workflow ทั่วไป
```bash
# 1. ตรวจสอบสถานะ
git status

# 2. เพิ่มไฟล์ที่เปลี่ยนแปลง
git add .

# 3. Commit การเปลี่ยนแปลง
git commit -m "ข้อความอธิบายการเปลี่ยนแปลง"

# 4. Push ขึ้น GitHub
git push
```

### ตัวอย่างการ Commit ที่ดี
```bash
# การแก้ไข bug
git commit -m "fix: Fix data preprocessing issue in app.py"

# การเพิ่ม feature ใหม่
git commit -m "feat: Add model evaluation metrics"

# การปรับปรุง documentation
git commit -m "docs: Update README with usage instructions"

# การ refactor code
git commit -m "refactor: Improve model loading efficiency"
```

---

## การจัดการ Branch

### สร้างและใช้งาน Branch
```bash
# ดู branch ทั้งหมด
git branch

# สร้าง branch ใหม่
git branch feature/new-model

# เปลี่ยนไป branch ใหม่
git checkout feature/new-model

# หรือสร้างและเปลี่ยนไปในคำสั่งเดียว
git checkout -b feature/new-model

# Push branch ใหม่ขึ้น GitHub
git push -u origin feature/new-model
```

### รวม Branch (Merge)
```bash
# กลับไป main branch
git checkout main

# รวม branch อื่นเข้ามา
git merge feature/new-model

# Push การเปลี่ยนแปลง
git push

# ลบ branch ที่ไม่ใช้แล้ว
git branch -d feature/new-model
```

---

## คำสั่ง Git ที่ควรรู้

### การดูข้อมูล
```bash
# ดูประวัติ commit
git log

# ดูประวัติแบบสั้น
git log --oneline

# ดูความแตกต่างของไฟล์
git diff

# ดูสถานะไฟล์
git status

# ดู remote repositories
git remote -v
```

### การยกเลิกการเปลี่ยนแปลง
```bash
# ยกเลิกการเปลี่ยนแปลงไฟล์ที่ยังไม่ commit
git checkout -- filename.py

# ยกเลิกการ add ไฟล์
git reset HEAD filename.py

# ยกเลิก commit ล่าสุด (แต่เก็บการเปลี่ยนแปลง)
git reset --soft HEAD~1

# ยกเลิก commit ล่าสุด (และลบการเปลี่ยนแปลง)
git reset --hard HEAD~1
```

### การดึงข้อมูลจาก GitHub
```bash
# ดึงข้อมูลล่าสุดจาก GitHub
git pull

# หรือแบบแยกขั้นตอน
git fetch
git merge origin/main
```

---

## Best Practices

### 1. การเขียน Commit Messages
- ใช้ Present tense: "Add feature" แทน "Added feature"
- ขึ้นต้นด้วย type: feat, fix, docs, refactor, test
- ข้อความสั้น ๆ แต่อธิบายชัดเจน
- หากมีรายละเอียดเยอะ ให้เขียนในบรรทัดใหม่

### 2. การจัดการไฟล์
- **อย่า** push model files ขนาดใหญ่ (ใช้ Git LFS หรือ cloud storage)
- **อย่า** push sensitive data (API keys, passwords)
- ใช้ .gitignore อย่างเหมาะสม
- เก็บเฉพาะ source code และ configuration files

### 3. การใช้ Branch
- ใช้ main/master สำหรับ production code
- สร้าง branch แยกสำหรับ feature ใหม่
- ตั้งชื่อ branch ให้สื่อความหมาย: `feature/model-improvement`

### 4. การ Backup
- Push บ่อย ๆ เพื่อป้องกันข้อมูลสูญหาย
- ทำ commit เล็ก ๆ ทีละนิด
- ใช้ tag สำหรับ version สำคัญ

---

## การแก้ปัญหาที่พบบ่อย

### 1. ไฟล์ขนาดใหญ่เกิน 100MB
```bash
# ลบไฟล์จาก git (แต่เก็บในเครื่อง)
git rm --cached large_file.bin

# เพิ่มลงใน .gitignore
echo "large_file.bin" >> .gitignore

# Commit การเปลี่ยนแปลง
git add .gitignore
git commit -m "Remove large file from tracking"
```

### 2. Merge Conflict
```bash
# เมื่อเกิด conflict ให้แก้ไขไฟล์ที่ขัดแย้ง
# หา <<<<<<< ======= >>>>>>> ในไฟล์แล้วแก้ไข

# หลังแก้ไขแล้ว
git add conflicted_file.py
git commit -m "Resolve merge conflict"
```

### 3. Push ถูก Reject
```bash
# ดึงข้อมูลล่าสุดก่อน
git pull --rebase origin main

# แล้ว push ใหม่
git push
```

### 4. ลืม .gitignore ไฟล์ที่ track แล้ว
```bash
# ลบจาก tracking (แต่เก็บไฟล์)
git rm -r --cached wcberta-prachathai67k/

# เพิ่มลง .gitignore
echo "wcberta-prachathai67k/" >> .gitignore

# Commit การเปลี่ยนแปลง
git add .
git commit -m "Remove model files from tracking"
```

---

## สำหรับโปรเจคนี้โดยเฉพาะ

### ไฟล์ที่ควร Push:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Dependencies
- ✅ `*.ipynb` - Jupyter notebooks (อาจจะ clear output ก่อน)
- ✅ `README.md` - Documentation
- ✅ Script files และ configuration files

### ไฟล์ที่ไม่ควร Push:
- ❌ `wcberta-prachathai67k/` - Model files (ขนาดใหญ่)
- ❌ `wcberta-prachathai67k-best/` - Model files (ขนาดใหญ่)
- ❌ `*.csv` - Data files (อาจมีขนาดใหญ่)
- ❌ `__pycache__/` - Python cache

### คำสั่งเริ่มต้นสำหรับโปรเจคนี้:
```bash
cd "d:\Python\HandOn_LLMs\wangchang_model_classify_new"
git init
git add app.py requirements.txt *.ipynb
git commit -m "Initial commit: Add WangChang news classification project"
git remote add origin https://github.com/YOUR_USERNAME/wangchang-news-classification.git
git push -u origin main
```

---

## 🚀 Quick Start Commands

สำหรับการเริ่มต้นใช้งานอย่างรวดเร็ว:

```bash
# 1. Setup Git (ครั้งเดียว)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 2. Initialize Repository
cd "d:\Python\HandOn_LLMs\wangchang_model_classify_new"
git init

# 3. Create .gitignore (สำคัญ!)
# สร้างไฟล์ .gitignore ตามที่แนะนำด้านบน

# 4. First Push
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main

# 5. Daily Workflow
git add .
git commit -m "Your commit message"
git push
```

---

**📝 หมายเหตุ**: 
- แทนที่ `YOUR_USERNAME` และ `YOUR_GITHUB_REPO_URL` ด้วยข้อมูลจริงของคุณ
- ก่อน push ครั้งแรก ให้ตรวจสอบขนาดไฟล์ด้วย `git ls-files --stage`
- หากมีปัญหา สามารถถามได้เสมอ!