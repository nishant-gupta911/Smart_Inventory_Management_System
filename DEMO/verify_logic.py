#!/usr/bin/env python3
"""
Manual test of Action logic to verify implementation
"""

# Test cases that should work with the corrected logic
test_cases = [
    # days_to_expiry, donation_eligible, expected_action
    (-10, False, "Remove"),      # Too expired
    (-3, True, "Donate"),        # Recently expired and eligible
    (-2, False, "Remove"),       # Recently expired but not eligible (should be Remove since < -5 is false, but -5 <= -2 <= -1 is true but not eligible)
    (0, False, "Apply Discount"), # Near expiry
    (3, False, "Apply Discount"), # Near expiry
    (5, False, "Apply Discount"), # Near expiry
    (10, False, "Restock"),      # Fresh
    (20, False, "Restock"),      # Fresh
]

print("Testing Action Logic:")
print("=" * 60)

for days, eligible, expected in test_cases:
    # Apply the logic
    if days < -5:
        actual = "Remove"
    elif -5 <= days <= -1 and eligible:
        actual = "Donate"
    elif 0 <= days <= 5:
        actual = "Apply Discount"
    else:
        actual = "Restock"
    
    status = "✅ PASS" if actual == expected else "❌ FAIL"
    print(f"Days: {days:3d} | Eligible: {eligible} | Expected: {expected:15s} | Actual: {actual:15s} | {status}")

print("\n" + "=" * 60)
print("Logic verification complete!")
