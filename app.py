import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical

# === CONFIG ===
SEQ_LENGTH = 4
FILE_PATH = 'library.csv'  # Ensure this CSV is in the same folder

# === Load dataset ===
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(FILE_PATH, on_bad_lines='skip')
    df.dropna(subset=['Branch'], inplace=True)

    branch_encoder = LabelEncoder()
    df['branch_encoded'] = branch_encoder.fit_transform(df['Branch'])

    branches = df['branch_encoded'].values
    sequences, next_branch = [], []

    for i in range(len(branches) - SEQ_LENGTH):
        sequences.append(branches[i:i + SEQ_LENGTH])
        next_branch.append(branches[i + SEQ_LENGTH])

    X = np.array(sequences)
    y = to_categorical(np.array(next_branch), num_classes=len(branch_encoder.classes_))

    return X, y, branch_encoder, df

# === Build LSTM model ===
def build_model(num_classes):
    model = Sequential([
        Embedding(input_dim=num_classes, output_dim=32, input_length=SEQ_LENGTH),
        LSTM(64),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# === UI Config ===
st.set_page_config(page_title="Library Branch Predictor", layout="centered")
st.title("📚 Library Branch Predictor")
st.markdown("Predict the next most likely library branch visit using either a trained LSTM model or historical data.")

# === Load Data & Train Model ===
X, y, branch_encoder, df = load_and_prepare_data()

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = build_model(len(branch_encoder.classes_))
with st.spinner("Training LSTM model..."):
    history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test), verbose=0)

# === Show accuracy ===
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

st.success("✅ LSTM model trained successfully!")
st.markdown(f"📈 **Training Accuracy:** `{train_acc * 100:.2f}%`")
st.markdown(f"🧪 **Testing Accuracy:** `{test_acc * 100:.2f}%`")


# === Tabbed Interface ===
tab1, tab2 = st.tabs(["🔮 Predict Using Sequence", "📊 Predict from Full Data"])

# === Tab 1: LSTM Prediction ===
with tab1:
    st.subheader("🧠 Sequence-based Prediction (LSTM)")

    branches_input = st.multiselect(
        f"Select the most recent {SEQ_LENGTH} branches (in order):",
        branch_encoder.classes_.tolist(),
        default=branch_encoder.classes_.tolist()[:SEQ_LENGTH]
    )

    if len(branches_input) != SEQ_LENGTH:
        st.warning(f"⚠️ Please select exactly {SEQ_LENGTH} branches.")
    else:
        try:
            recent_encoded = branch_encoder.transform(branches_input)
            sequence = np.array(recent_encoded).reshape(1, -1)
            prediction = model.predict(sequence, verbose=0)[0]
            predicted_branch = branch_encoder.inverse_transform([np.argmax(prediction)])[0]

            # Show result
            st.success(f"🔮 Predicted Next Branch: `{predicted_branch}`")

            # Show confidence bar
            st.subheader("📊 Prediction Confidence")
            fig, ax = plt.subplots()
            ax.bar(branch_encoder.classes_, prediction, color='mediumseagreen')
            ax.set_ylabel("Probability")
            ax.set_xlabel("Branch")
            ax.set_ylim(0, 1)
            ax.set_title("Confidence for Each Branch")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# === Tab 2: Frequency-based Prediction ===
with tab2:
    st.subheader("📈 Historical Most Frequent Branch")
    most_common = df['Branch'].value_counts().idxmax()
    count = df['Branch'].value_counts().max()
    st.info(f"🏆 The most frequently visited branch is **`{most_common}`** with **{count}** visits.")

    st.markdown("### 🔢 Branch Visit Frequencies")
    fig2, ax2 = plt.subplots()
    df['Branch'].value_counts().plot(kind='bar', color='slateblue', ax=ax2)
    ax2.set_ylabel("Visit Count")
    ax2.set_title("Branch Visit Frequency")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
