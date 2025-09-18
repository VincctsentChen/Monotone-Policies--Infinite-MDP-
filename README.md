Infinite-Horizon MDP for Disease Progression

This repo builds an infinite-horizon Markov Decision Process (MDP) to model patients’ disease progression under different treatments.
The transition tensor has shape (10, 10, 6): 10 states × 10 next-states × 6 actions.


# Run the test script (produces the transition tensor and logs)
python test_infinite_horizon_with_data.py

After running the script, you can quickly validate the transition shape:

# Example import — adjust to your actual function/location as needed:
# from transition_probabilities_infinite import build_transition_tensor

# For demonstration, replace the next line with the real function that returns your tensor.
# T = build_transition_tensor()  # expected shape (10, 10, 6)

# If your test script saves/returns the tensor, import or load it here instead.
# For now, this just shows the expected shape check:
expected_shape = (10, 10, 6)
print("Expected transition tensor shape:", expected_shape)
# print("Actual shape:", T.shape); assert T.shape == expected_shape
PY

Actions / Treatments

Defined in test_infinite_horizon_with_data.py:

T0: "NT" – No Treatment

T1: "ACE" – ACE Inhibitors

T2: "ARB" – Angiotensin Receptor Blockers

T3: "BB" – Beta Blockers

T4: "CCB" – Calcium Channel Blockers

T5: "TH" – Thiazide Diuretics

# in test_infinite_horizon_with_data.py
alldrugs = ["ACE", "ARB", "BB", "CCB", "TH"]  # base list
drugs.insert(0, "NT")                          # add no-treatment
alldrugs = drugs                               # simplified: single-drug actions

Run the MDP test
python test_infinite_horizon_with_data.py



