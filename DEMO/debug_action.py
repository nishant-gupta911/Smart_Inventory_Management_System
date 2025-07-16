#!/usr/bin/env python3
"""
Debug script to check Action logic
"""

# Test the logic for the problematic case
days_to_expiry = 17
donation_eligible = False

print(f"Testing: days_to_expiry = {days_to_expiry}, donation_eligible = {donation_eligible}")

if days_to_expiry < -5:
    action = 'Remove'
    print("Action: Remove (days_to_expiry < -5)")
elif -5 <= days_to_expiry <= -1 and donation_eligible:
    action = 'Donate'
    print("Action: Donate (-5 <= days_to_expiry <= -1 and donation_eligible)")
elif 0 <= days_to_expiry <= 5:
    action = 'Apply Discount'
    print("Action: Apply Discount (0 <= days_to_expiry <= 5)")
else:
    action = 'Restock'
    print("Action: Restock (else case)")

print(f"Final action: {action}")

# Test conditions individually
print(f"\nCondition checks:")
print(f"days_to_expiry < -5: {days_to_expiry < -5}")
print(f"-5 <= days_to_expiry <= -1: {-5 <= days_to_expiry <= -1}")
print(f"donation_eligible: {donation_eligible}")
print(f"(-5 <= days_to_expiry <= -1) and donation_eligible: {(-5 <= days_to_expiry <= -1) and donation_eligible}")
print(f"0 <= days_to_expiry <= 5: {0 <= days_to_expiry <= 5}")
