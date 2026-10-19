from pathlib import Path
import pandas as pd, streamlit as st, matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="BinBuddy AI",page_icon="♻️",layout="wide")
st.markdown("""<style>
.stApp{background:#fff;color:#183326}
[data-testid="stHeader"]{background:#fff}
[data-testid="stMarkdownContainer"] p,label,[data-testid="stCaptionContainer"]{color:#183326}
h1,h2,h3,h4{color:#14532d!important}
div[data-testid="stMetric"]{background:#eaf4ed;border:1px solid #b8d9c0;padding:14px;border-radius:12px}
div[data-testid="stMetric"] label,div[data-testid="stMetric"] [data-testid="stMetricLabel"] p{color:#234b32!important}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#14532d!important}
div.stButton>button,div.stFormSubmitButton>button{background:#15803d;color:white;border:0;border-radius:9px;font-weight:700}
div.stButton>button:hover,div.stFormSubmitButton>button:hover{background:#14532d;color:white}
.hero{padding:1.4rem 2rem;border-radius:18px;background:linear-gradient(110deg,#14532d,#15803d);color:white;margin:.4rem 0 1rem}
.hero h1{color:white!important;font-size:2.5rem}.hero p{color:white!important}
.note{background:#eaf4ed;color:#183326;border-left:5px solid #15803d;padding:.8rem;border-radius:8px}
div[data-testid="stRadio"] label p{color:#183326!important}
div[data-testid="stExpander"] summary p{color:#14532d!important;font-weight:700}
input,textarea{color:#183326!important}
</style>""",unsafe_allow_html=True)
@st.cache_data
def load_data(): return pd.read_csv(Path(__file__).with_name("waste_dataset.csv"))
@st.cache_resource
def train():
 d=load_data(); X,y=d[["material","food_or_liquid_soiled","broken_or_damaged"]],d["waste_category"]
 X1,X2,y1,y2=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
 p=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),X.columns.tolist())])
 m=Pipeline([("prep",p),("tree",DecisionTreeClassifier(max_depth=7,random_state=42))])
 m.fit(X1,y1); return m,X2,y2,m.predict(X2)
d=load_data(); model,xt,yt,yp=train()
items={
"Newspaper":"paper","Magazine":"paper","Office paper":"paper","Envelope":"paper","Cardboard box":"paper","Paper bag":"paper","Egg carton":"paper","Paper carton":"paper",
"Water bottle":"plastic","Plastic bag":"plastic","Plastic wrapper":"plastic","Food container":"plastic","Yogurt tub":"plastic","Plastic lid":"plastic","Shampoo bottle":"plastic","Plastic toy":"plastic",
"Glass bottle":"glass","Glass jar":"glass","Glass cosmetics bottle":"glass","Aluminium can":"metal","Steel tin":"metal","Metal lid":"metal","Scrap metal":"metal","Aluminium foil":"metal",
"Food scraps":"food","Fruit peels":"food","Vegetable peels":"food","Leftover rice":"food","Tea leaves / coffee grounds":"food","Eggshells":"food",
"Mobile phone":"electronic","Battery":"electronic","Charger / USB cable":"electronic","Keyboard / mouse":"electronic","Earphones":"electronic","Small appliance":"electronic",
"Old T-shirt":"textile","Jeans":"textile","Towel / bedsheet":"textile","Fabric scraps":"textile","Socks":"textile","Wood offcuts":"paper","Rubber item":"plastic","Ceramic mug / plate":"glass","Garden leaves":"food"}
st.markdown('<div class="hero"><h1>♻️ BinBuddy AI</h1><p>Your smart companion for more informed waste sorting.</p><p>SDG 12 · Responsible Consumption and Production</p></div>',unsafe_allow_html=True)
page=st.radio("Navigation",["Home Dashboard","Waste Classifier","Dataset Explorer","Model Evaluation","Disposal Guide","About Project"],horizontal=True,label_visibility="collapsed")
if page=="Home Dashboard":
 st.subheader("Welcome to BinBuddy AI"); st.write("A school AI prototype exploring waste sorting and responsible consumption.")
 a,b,c=st.columns(3); a.metric("Dataset records",f"{len(d):,}"); b.metric("Waste categories",d.waste_category.nunique()); c.metric("Test-set accuracy",f"{accuracy_score(yt,yp):.1%}")
 st.subheader("Waste categories in the sample dataset"); st.bar_chart(d.waste_category.value_counts())
 st.markdown('<div class="note"><b>Prototype note:</b> The expanded dataset includes synthetic examples. Test accuracy is not proof of real-world performance.</div>',unsafe_allow_html=True)
elif page=="Waste Classifier":
 st.subheader("♻️ Waste Classifier")
 search_item=st.text_input("🔎 Search for a waste item",placeholder="Type e.g. bottle, battery, newspaper, jeans...")
 matches=[name for name in items if search_item.strip().lower() in name.lower()]
 if search_item and not matches: st.warning("No matching item found. Try a broader search term.")
 with st.form("classify"):
  item=st.selectbox("What item are you sorting?",matches if matches else list(items))
  soil=st.radio("Is it soiled with food or liquid?",["no","yes"],horizontal=True,format_func=lambda x:"No" if x=="no" else "Yes")
  broken=st.radio("Is it broken or damaged?",["no","yes"],horizontal=True,format_func=lambda x:"No" if x=="no" else "Yes")
  go=st.form_submit_button("Predict waste category",use_container_width=True)
 if go:
  pred=model.predict(pd.DataFrame([dict(material=items[item],food_or_liquid_soiled=soil,broken_or_damaged=broken)]))[0]
  st.success(f"{item} → {pred}")
  guides={"E-waste":"Use an authorised e-waste collection point.","Biodegradable":"Compost only where suitable local facilities/rules allow.","Recyclable":"Check local acceptance rules; empty and rinse where required.","Reusable/ textile collection":"Consider repair, reuse, donation, or textile collection.","General waste (check local rules)":"Check local council guidance."}
  st.info(guides.get(pred,"Check local disposal guidance."))
  if item in ["Battery","Mobile phone","Charger / USB cable","Small appliance"]: st.warning("Use specialist collection guidance for batteries and electronics.")
 st.caption("Items are mapped to broad material groups. This educational model can make mistakes.")
elif page=="Dataset Explorer":
 st.subheader("Dataset Explorer"); q=st.text_input("Search dataset")
 view=d if not q else d[d.astype(str).apply(lambda c:c.str.contains(q,case=False,na=False)).any(axis=1)]
 st.dataframe(view,use_container_width=True,hide_index=True); st.caption(f"Showing {len(view)} of {len(d)} records."); st.bar_chart(d.waste_category.value_counts())
elif page=="Model Evaluation":
 st.subheader("Model Evaluation"); st.metric("Held-out test accuracy",f"{accuracy_score(yt,yp):.1%}")
 st.warning("This is a small illustrative dataset with synthetic rows; do not claim the score represents real-world performance.")
 st.dataframe(pd.DataFrame(classification_report(yt,yp,output_dict=True,zero_division=0)).transpose().round(3),use_container_width=True)
 labs=sorted(d.waste_category.unique()); cm=confusion_matrix(yt,yp,labels=labs); fig,ax=plt.subplots(figsize=(10,5)); ConfusionMatrixDisplay(cm,display_labels=labs).plot(ax=ax,xticks_rotation=35,colorbar=False); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
elif page=="Disposal Guide":
 st.subheader("🗑️ Disposal Guide")
 st.write("General guidance only. Local municipal and collection-provider rules take priority.")
 guide=[
 ("Paper and cardboard","Usually accepted where local services allow: clean, dry newspapers, office paper, magazines, envelopes, and cardboard. Flatten boxes and remove plastic wrap or tape where practical. Keep paper dry. Greasy or food-soiled paper, tissues, laminated paper, and coated cups may not be accepted because they can contaminate the recycling process. Check the local programme before placing cartons or special papers in the paper stream."),
 ("Plastic containers and packaging","Examples include drink bottles, detergent bottles, tubs, trays, bags, wrappers, and takeaway containers. Empty containers and rinse them if your service asks. A recycling symbol or resin number does not guarantee local acceptance. Flexible films, multi-layer packets, foam, black plastic, and heavily contaminated packaging may need a different route. Reduce and reuse packaging where practical."),
 ("Glass bottles and jars","Bottles and food jars may be accepted by local glass collection. Empty them and follow local instructions for lids. Mirrors, window glass, drinking glasses, ceramics, bulbs, and heat-resistant cookware may have different compositions and should not automatically go with bottle glass. Broken glass can injure people: do not pick up shards with bare hands; follow local safe-wrapping or special-collection instructions."),
 ("Metal cans, foil and scrap","Aluminium drink cans and steel food tins may be accepted; foil, lids, cookware, wires, and scrap metal vary by programme. Empty containers and check whether foil should be balled up or kept separate. Handle sharp edges carefully. Aerosols, pressurised containers, and metal items contaminated with hazardous substances may require special handling."),
 ("Food scraps and garden waste","Fruit and vegetable peels, coffee grounds, tea leaves, eggshells, leaves, and small garden trimmings may be compostable where facilities exist. Home compost systems and municipal organics services can accept different items. Meat, fish, dairy, oils, cooked food, pet waste, and diseased plants may be restricted, so check the compost provider's rules. Keep food waste separate from dry recyclables to reduce odours and contamination."),
 ("Electronic waste (e-waste)","Phones, computers, chargers, cables, keyboards, earphones, and small appliances should go to authorised collection, take-back, or approved recycling schemes. Do not dismantle electronics casually because they may contain hazardous components. Before donating or recycling a working device, back up and remove personal data where possible. Keep e-waste out of ordinary household bins and mixed recycling."),
 ("Batteries","Batteries can cause fires or leak hazardous substances if crushed, damaged, or incorrectly sorted. Use designated battery collection points or authorised take-back services. Follow local advice for safe storage and transport; keep terminals from touching metal objects when recommended. Never open, heat, crush, or puncture a battery. For a swollen, leaking, hot, or damaged battery, avoid handling it and seek specialist local advice."),
 ("Clothing, shoes and textiles","Repair, reuse, swap, or donate clean wearable items first. Textile collection schemes may accept damaged clothing for fibre recovery, but rules differ. Keep donations clean and dry; wet, mouldy, oily, or contaminated textiles may be rejected. Shoes, mattresses, carpets, and mixed-material bags can have separate collection routes."),
 ("Wood, rubber, ceramics and mixed materials","Wood offcuts, rubber products, ceramic mugs, mirrors, composite packaging, and mixed-material objects are not automatically accepted in standard recycling. Reuse or repair where safe; check municipal bulky-waste, construction-waste, or specialist drop-off services. Avoid burning treated wood or unknown materials. Separate components only if safe and local guidance recommends it."),
 ("General, contaminated or unknown waste","Use residual waste only after checking whether reuse, recycling, composting, or specialist collection is available. Food-soiled packaging and mixed-material products may not be accepted in recycling. Chemicals, paints, solvents, medicines, medical sharps, fluorescent lamps, and other hazardous items should not go into ordinary bins without official disposal guidance. Contact your municipal waste service if unsure.")
 ]
 for title,body in guide:
  with st.expander(title):
   st.write(body)
 st.markdown('<div class="note"><b>Safety reminder:</b> Keep batteries, chemicals, medical sharps, and unknown hazardous materials out of ordinary recycling. Use authorised specialist collection routes.</div>',unsafe_allow_html=True)
else:
 st.subheader("About BinBuddy AI")
 st.write("BinBuddy AI is a Class 12 Artificial Intelligence project demonstrating supervised machine learning for waste-sorting education. It connects to UN Sustainable Development Goal 12: Responsible Consumption and Production.")
 st.markdown("### Problem statement"); st.write("Waste is often mixed because people are unsure how to sort items. Incorrect sorting can contaminate recyclable material. This project demonstrates a simple digital learning aid.")
 st.markdown("### Aim and objectives"); st.markdown("- Demonstrate a supervised-learning workflow.\\n- Use material, contamination, and damage as features.\\n- Present predictions in an accessible interface.\\n- Explore data and evaluate model outputs.\\n- Encourage responsible disposal awareness.")
 st.markdown("### Methodology"); st.markdown("1. Pandas loads the CSV dataset.\\n2. OneHotEncoder transforms categorical features.\\n3. A Decision Tree Classifier learns from labelled examples.\\n4. A held-out test split is evaluated with accuracy, classification report, and confusion matrix.\\n5. Streamlit provides the website interface.\\n6. Orange Data Mining is used for a separate visual workflow on the same CSV.")
 st.markdown("### Technologies"); st.write("Python · Pandas · scikit-learn · Streamlit · Matplotlib · Orange Data Mining")
 st.markdown("### Limitations"); st.write("This is an educational prototype, not a certified waste-management tool. The dataset is small and includes synthetic examples. The model cannot inspect photos, identify chemical composition, or know local rules. Predictions may be wrong. A real system needs a larger, independently collected and expert-checked dataset.")
 st.markdown("### Expected impact"); st.write("The project aims to build awareness of waste categories and demonstrate an end-to-end AI workflow that encourages reuse, recycling, composting, and responsible consumption.")
 st.markdown("### Responsible use"); st.write("Treat predictions as educational suggestions. Follow local rules and authorised collection routes for e-waste, batteries, and hazardous items.")
