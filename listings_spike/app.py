# ============ Data Input ============
# Use the CSV that lives in the repo root:
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "blackpool_low_epc_with_coords.csv"
)

st.markdown("**Data source:** Built-in CSV (Blackpool EPC). You can still upload a file to override it.")

uploaded = st.file_uploader(
    "Optional: Upload a different EPC CSV "
    "(ADDRESS, POSTCODE, LAT, LON, EPC_CURRENT, EPC_POTENTIAL, UNITS_PER_BUILDING)",
    type=["csv"]
)

if uploaded:
    df = load_epc_csv(uploaded)
else:
    df = load_epc_csv(DATA_PATH)
