import json, torch, numpy as np
import traceback
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Thai News Classification",
    page_icon="📰",
    layout="centered"
)

MODEL_DIR = "wcberta-prachathai67k-best"
MAX_LEN = 512
THRESH = 0.5   

# กำหนด device ก่อน
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize session state for model
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.labels = None
    st.session_state.error_message = ""

def load_model_and_tokenizer():
    """โหลด model และ tokenizer พร้อม error handling"""
    try:
        print(f"Loading tokenizer from {MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        print(f"Loading model from {MODEL_DIR}...")
        # ลองวิธีต่างๆ ในการโหลด model
        model = None
        
        # วิธีที่ 1: โหลดแบบปกติ
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_DIR, 
                dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=False
            )
            model = model.to(device)
            model.eval()
            print("✅ Model loaded successfully with method 1")
            
        except Exception as e1:
            print(f"❌ Method 1 failed: {e1}")
            
            # วิธีที่ 2: ใช้ to_empty()
            try:
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
                if hasattr(model, 'to_empty'):
                    model = model.to_empty(device=device)
                else:
                    model = model.to(device)
                model.eval()
                print("✅ Model loaded successfully with method 2")
                
            except Exception as e2:
                print(f"❌ Method 2 failed: {e2}")
                
                # วิธีที่ 3: โหลดใน CPU ก่อน
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        MODEL_DIR,
                        dtype=torch.float32
                    )
                    model = model.to(device)
                    model.eval()
                    print("✅ Model loaded successfully with method 3")
                except Exception as e3:
                    print(f"❌ All methods failed: {e3}")
                    raise e3
        
        # โหลด labels
        with open(f"{MODEL_DIR}/label_names.json", encoding="utf-8") as f:
            labels = json.load(f)
        
        print(f" ✅ All components loaded successfully! ")
        print(f"📊 Device: {device}")
        print(f"📚 Vocab size: {tokenizer.vocab_size}")
        print(f"🏷️ Number of labels: {len(labels)}")
        
        return model, tokenizer, labels, ""
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return None, None, None, error_msg

def predict_with_probs(texts: list[str], model, tokenizer, labels, threshold: float = THRESH, top_k: int = None):
    """ทำนายผลพร้อม detailed error reporting"""
    
    # ตรวจสอบว่า model ถูกโหลดแล้ว
    if model is None or tokenizer is None:
        error_msg = "Model หรือ tokenizer ไม่ได้ถูกโหลด"
        print(f"❌ {error_msg}")
        return [{"text": text, "probs_sorted": [], "chosen": [], "error": error_msg} for text in texts]
    
    try:
        print(f"🔤 Input texts: {texts}")
        print(f"📏 Text lengths: {[len(text) for text in texts]}")
        
        # Tokenize
        enc = tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_LEN, 
            padding=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        
        print(f"🔢 Token shape: {enc['input_ids'].shape}")
        print(f"📊 Token range: {enc['input_ids'].min()} - {enc['input_ids'].max()}")
        print(f"📚 Vocab size: {tokenizer.vocab_size}")
        
        # ตรวจสอบ token IDs
        vocab_size = tokenizer.vocab_size
        if torch.any(enc['input_ids'] >= vocab_size):
            print(f"⚠️ WARNING: Found token IDs >= vocab_size ({vocab_size})")
            enc['input_ids'] = torch.clamp(enc['input_ids'], 0, vocab_size - 1)
            
        if torch.any(enc['input_ids'] < 0):
            print("⚠️ WARNING: Found negative token IDs")
            enc['input_ids'] = torch.clamp(enc['input_ids'], 0, vocab_size - 1)
        
        # Move to device
        enc = {k: v.to(device) for k, v in enc.items()}
        
        # Model prediction
        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            print("🤖 Running model prediction...")
            outputs = model(**enc)
            logits = outputs.logits
            probs = sigmoid(logits).cpu().numpy()
            print(f"✅ Prediction successful! Output shape: {probs.shape}")
            
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return [{"text": text, "probs_sorted": [], "chosen": [], "error": error_msg} for text in texts]

    # สร้างผลลัพธ์
    results = []
    for i, text in enumerate(texts):
        try:
            p = probs[i]
            order = np.argsort(p)[::-1]
            probs_sorted = [(labels[j], float(p[j])) for j in order]

            if top_k and top_k > 0:
                chosen = [labels[j] for j in order[:top_k]]
            else:
                chosen = []
                for name, prob in probs_sorted:
                    if prob >= threshold:
                        chosen.append(name)
                    else:
                        break

            results.append({
                "text": text,
                "probs_sorted": probs_sorted,
                "chosen": chosen,
                "error": None
            })
            
        except Exception as e:
            error_msg = f"Result processing error for text {i}: {str(e)}"
            print(f"❌ {error_msg}")
            results.append({
                "text": text,
                "probs_sorted": [],
                "chosen": [],
                "error": error_msg
            })
            
    print(f"✅ Generated {len(results)} results")
    return results

# โหลด model เมื่อเริ่มต้น
if not st.session_state.model_loaded:
    with st.spinner("🚀 กำลังโหลดโมเดล..."):
        model, tokenizer, labels, error = load_model_and_tokenizer()
        
        if model is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.labels = labels
            st.session_state.model_loaded = True
            st.session_state.error_message = ""
            st.success("✅ โมเดลโหลดสำเร็จ!")
        else:
            st.session_state.error_message = error
            st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {error}")

# UI
st.title("📰 Thai News Classification")
st.markdown("แอปพลิเคชันสำหรับจำแนกประเภทข่าวภาษาไทย โดยใช้โมเดล `wcberta-prachathai67k`")

# แสดงสถานะ model
if st.session_state.model_loaded:
    st.success("🟢 โมเดลพร้อมใช้งาน")
else:
    st.error("🔴 โมเดลไม่พร้อมใช้งาน")
    if st.session_state.error_message:
        with st.expander("ดูรายละเอียด Error"):
            st.text(st.session_state.error_message)

# Debug information
with st.expander("🔧 ข้อมูล Debug"):
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Device:** {device}")
        st.write(f"**Model Loaded:** {st.session_state.model_loaded}")
    with col2:
        if st.session_state.tokenizer:
            st.write(f"**Vocab Size:** {st.session_state.tokenizer.vocab_size}")
        if st.session_state.labels:
            st.write(f"**Labels:** {len(st.session_state.labels)}")

st.write("ป้อนข้อความที่ต้องการทำนาย:")
input_texts = st.text_area("ใส่ข้อความที่นี่:", height=200, 
                          placeholder="ตัวอย่าง: กรมอุตุฯ เตือนวันนี้ทั่วไทยเจอฝนถล่ม...")

if st.button("🧠 ทำนายผล"):
    if not st.session_state.model_loaded:
        st.error("โมเดลไม่ได้ถูกโหลด กรุณารีเฟรชหน้าเพื่อโหลดใหม่")
    elif not input_texts.strip():
        st.warning("กรุณาป้อนข้อความเพื่อทำนายผล")
    else:
        valid_texts = [input_texts.strip()]
        
        with st.spinner("🤖 กำลังวิเคราะห์ข้อความ..."):
            results = predict_with_probs(
                valid_texts, 
                st.session_state.model,
                st.session_state.tokenizer,
                st.session_state.labels,
                threshold=None, 
                top_k=3
            )
            
            # Debug: แสดงผลลัพธ์
            st.markdown(f"**Debug Results:** `{len(results)}` items")
            for i, result in enumerate(results):
                st.markdown(f"Result {i}: {len(result.get('probs_sorted', []))} probabilities")
            
            if results and any(result.get('probs_sorted') for result in results):
                st.success("วิเคราะห์สำเร็จ!")
                
                for result in results:
                    if result.get('error'):
                        st.error(f"Error: {result['error']}")
                        continue
                        
                    with st.container(border=True):
                        st.markdown(f"**ข้อความ:** {result['text']}")
                        
                        if result['chosen']:
                            tags = ' '.join([f"`{cat}`" for cat in result['chosen']])
                            st.markdown(f"**หมวดหมู่ที่ทำนาย:** {tags}")
                        else:
                            st.markdown("**หมวดหมู่ที่ทำนาย:** ไม่สามารถจำแนกได้")

                        with st.expander("ดูความน่าจะเป็นทั้งหมด"):
                            if result['probs_sorted']:
                                df = pd.DataFrame(result['probs_sorted'], columns=['หมวดหมู่', 'ความน่าจะเป็น'])
                                st.dataframe(df)
                            else:
                                st.write("ไม่มีข้อมูลความน่าจะเป็น")
            else:
                st.error("เกิดข้อผิดพลาดในการทำนาย หรือไม่มีผลลัพธ์")
                
                # แสดงข้อมูล debug เมื่อเกิดปัญหา
                if results:
                    for i, result in enumerate(results):
                        if result.get('error'):
                            st.text(f"Error in result {i}: {result['error']}")

# ปุ่มรีเซ็ต model
if st.button("🔄 รีโหลดโมเดล"):
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.labels = None
    st.experimental_rerun()