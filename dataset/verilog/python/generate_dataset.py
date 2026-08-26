import pandas as pd
import random
import os

data = []

for i in range(500):
    A = random.randint(0, 1)
    B = random.randint(0, 1)
    Cin = random.randint(0, 1)

    # Full Adder outputs
    Sum = A ^ B ^ Cin
    Cout = (A & B) | (B & Cin) | (A & Cin)

    # Simulated IC parameters
    switching_activity = A + B + Cin
    transistor_count = 28

    # Simulated power consumption (mW)
    power = (
        0.5
        + switching_activity * 0.8
        + random.uniform(0.0, 0.5)
    )

    # Simulated propagation delay (ns)
    delay = (
        1.0
        + switching_activity * 0.25
        + random.uniform(0.0, 0.2)
    )

    data.append([
        A, B, Cin, Sum, Cout,
        switching_activity,
        transistor_count,
        power,
        delay
    ])

columns = [
    "A", "B", "Cin", "Sum", "Cout",
    "Switching_Activity",
    "Transistor_Count",
    "Power_mW",
    "Delay_ns"
]

df = pd.DataFrame(data, columns=columns)

output_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "dataset",
    "power_delay_dataset.csv"
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_csv(output_path, index=False)

print("Dataset generated successfully!")
print(f"Total samples: {len(df)}")
print(f"Saved at: {output_path}")