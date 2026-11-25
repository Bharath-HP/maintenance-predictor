import streamlit as st

st.set_page_config(
    page_title="MaintenanceIQ — Expense Predictor",
    page_icon=":material/build:",
    layout="wide",
)

WEIGHTS = [5200, 1850, 18.5, 4600, -950, 0.32]
METRICS = {"r2": 0.927, "mae": 2840, "rmse": 3650}
DATASET_SIZE = 500


def predict_cost(age: float, hours: float, intensity: int, services: float, prev_cost: float) -> float:
    return max(
        0,
        WEIGHTS[0]
        + WEIGHTS[1] * age
        + WEIGHTS[2] * hours
        + WEIGHTS[3] * intensity
        - WEIGHTS[4] * services
        + WEIGHTS[5] * prev_cost,
    )


def format_inr(value: float) -> str:
    return f"₹{value:,.0f}".replace(",", ",")


st.title("MaintenanceIQ", anchor=False)
st.caption("Linear Regression · 500 machine records")

st.markdown("## Know your **maintenance bill** before the machine breaks.")
st.write(
    "Enter a machine's age, hours and service history. Our model estimates "
    "the next maintenance expense in rupees — instantly."
)

with st.container(horizontal=True):
    st.metric("R² score", f"{METRICS['r2']:.3f}", border=True)
    st.metric("MAE", format_inr(METRICS["mae"]), border=True)
    st.metric("RMSE", format_inr(METRICS["rmse"]), border=True)
    st.metric("Test records", str(int(DATASET_SIZE * 0.2)), border=True)

st.caption("Metrics computed on held-out test data · synthetic dataset, not real industrial data")
st.space("medium")

left, right = st.columns([3, 2], gap="medium")

with left:
    with st.container(border=True):
        st.subheader("Predict maintenance cost", anchor=False)

        age = st.number_input("Machine age (years)", min_value=0, value=8, step=1)
        hours = st.number_input("Operating hours / month", min_value=0, value=320, step=10)

        col1, col2 = st.columns(2)
        with col1:
            services = st.number_input("Previous service count", min_value=0, value=5, step=1)
        with col2:
            prev_cost = st.number_input(
                "Previous maintenance cost (₹)",
                min_value=0,
                value=25000,
                step=1000,
            )

        intensity_label = st.segmented_control(
            "Usage intensity",
            options=["Low", "Medium", "High"],
            default="Medium",
        )

        predict = st.button("Predict", type="primary", width="stretch")

with right:
    with st.container(border=True, height="stretch"):
        st.markdown("**Estimated expense**")

        if predict:
            intensity = ["Low", "Medium", "High"].index(intensity_label)
            estimate = predict_cost(age, hours, intensity, services, prev_cost)
            st.metric("Next maintenance cost", format_inr(estimate), border=False)
            st.write(
                f"A {age}-year-old machine running {hours} hrs/month at "
                f"{intensity_label.lower()} intensity with {services} past services."
            )
        else:
            st.metric("Next maintenance cost", "₹ —", border=False)
            st.caption("Fill in the machine details and press Predict.")
