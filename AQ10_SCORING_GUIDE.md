# AQ-10 Scoring Verification Guide

## ✅ Correct Implementation Summary

### Forward-Scored Questions (YES = Risk = 1)

| Q# | Question | YES means... | Score |
|----|----------|-------------|-------|
| **A1** | Does the child often notice small sounds when others do not? | Hyper-focus on details (autism trait) | 1 |
| **A7** | When read a story, does the child find it difficult to work out the character's intentions or feelings? | Difficulty with theory of mind | 1 |
| **A10** | Does the child find it hard to make new friends? | Social difficulty | 1 |

### Reverse-Scored Questions (NO = Risk = 1)

| Q# | Question | NO means... | Score |
|----|----------|------------|-------|
| **A2** | Does the child usually concentrate more on the whole picture rather than the small details? | Over-focuses on details instead | 1 |
| **A3** | In a social group, can the child easily keep track of several different people's conversations? | Cannot track multiple conversations | 1 |
| **A4** | Does the child find it easy to go back and forth between different activities? | Rigid, difficulty with transitions | 1 |
| **A5** | Does the child know how to keep a conversation going with his/her peers? | Poor conversational skills | 1 |
| **A6** | Is the child good at social chit-chat? | Poor social communication | 1 |
| **A8** | When he/she was in preschool, did he/she use to enjoy playing games involving pretending with other children? | Lack of pretend/imaginative play | 1 |
| **A9** | Does the child find it easy to work out what someone is thinking or feeling just by looking at their face? | Difficulty reading emotions | 1 |

---

## 🧪 Test Cases

### Test Case 1: Neurotypical Child (Low Risk)

**Expected Answers:**
- A1: NO (doesn't hyper-focus on sounds) → Score 0
- A2: YES (sees whole picture) → Score 0
- A3: YES (tracks conversations) → Score 0
- A4: YES (easy transitions) → Score 0
- A5: YES (good conversations) → Score 0
- A6: YES (good at chit-chat) → Score 0
- A7: NO (understands emotions in stories) → Score 0
- A8: YES (enjoyed pretend play) → Score 0
- A9: YES (reads faces well) → Score 0
- A10: NO (makes friends easily) → Score 0

**AQ-10 Score: 0/10** → **LOW RISK** ✅

---

### Test Case 2: Child with Autism Traits (High Risk)

**Expected Answers:**
- A1: YES (hyper-sensitive to sounds) → Score 1 ⚠️
- A2: NO (focuses on details, not whole) → Score 1 ⚠️
- A3: NO (can't track multiple conversations) → Score 1 ⚠️
- A4: NO (difficulty switching activities) → Score 1 ⚠️
- A5: NO (struggles with conversations) → Score 1 ⚠️
- A6: NO (poor at social chit-chat) → Score 1 ⚠️
- A7: YES (difficulty understanding emotions) → Score 1 ⚠️
- A8: NO (didn't enjoy pretend play) → Score 1 ⚠️
- A9: NO (can't read emotions from faces) → Score 1 ⚠️
- A10: YES (hard to make friends) → Score 1 ⚠️

**AQ-10 Score: 10/10** → **HIGH RISK** ⚠️

---

### Test Case 3: Borderline (Moderate Risk)

**Expected Answers:**
- A1: NO → Score 0
- A2: YES → Score 0
- A3: NO → Score 1 ⚠️
- A4: NO → Score 1 ⚠️
- A5: NO → Score 1 ⚠️
- A6: NO → Score 1 ⚠️
- A7: YES → Score 1 ⚠️
- A8: YES → Score 0
- A9: NO → Score 1 ⚠️
- A10: NO → Score 0

**AQ-10 Score: 6/10** → **MODERATE RISK** ⚠️

---

## 📊 Scoring Interpretation

| AQ-10 Score | Risk Level | Recommendation |
|-------------|-----------|----------------|
| **0-3** | Low Risk | Typical development likely |
| **4-5** | Mild Risk | Monitor development, consider follow-up |
| **6-7** | Moderate Risk | Recommend professional evaluation |
| **8-10** | High Risk | Strongly recommend comprehensive autism assessment |

**Clinical Cutoff:** Typically **≥6** indicates need for further evaluation

---

## 🔍 Code Logic Verification

```python
# Forward scored questions (YES = 1)
forward_scored = ['A1_Score', 'A7_Score', 'A10_Score']

# Scoring logic
if key in forward_scored:
    # A1, A7, A10: YES = risk
    score = 1 if answer == "Yes" else 0
else:
    # A2, A3, A4, A5, A6, A8, A9: NO = risk
    score = 1 if answer == "No" else 0
```

### Example Walkthrough:

**Question A1:** "Does the child often notice small sounds when others do not?"
- User answers: **YES**
- Logic: A1 is in `forward_scored` list
- Score: `1 if "Yes" == "Yes" else 0` → **1** ✅ (Risk indicator)

**Question A2:** "Does the child usually concentrate more on the whole picture...?"
- User answers: **NO**
- Logic: A2 is NOT in `forward_scored` list (reverse scored)
- Score: `1 if "No" == "No" else 0` → **1** ✅ (Risk indicator)

**Question A5:** "Does the child know how to keep a conversation going...?"
- User answers: **YES**
- Logic: A5 is NOT in `forward_scored` list (reverse scored)
- Score: `1 if "Yes" == "No" else 0` → **0** ✅ (No risk)

---

## ✅ Implementation Checklist

- [x] All 10 AQ-10 questions with exact wording
- [x] Forward scoring for A1, A7, A10
- [x] Reverse scoring for A2, A3, A4, A5, A6, A8, A9
- [x] Correct demographic encoding (Gender, Jaundice, Autism, Used_app)
- [x] One-hot encoding for Ethnicity, Country, Relation
- [x] Feature order matches training data
- [x] Scaling applied when needed
- [x] Clear UI instructions
- [x] Score interpretation guidance

---

## 🎯 Final Status

**All scoring logic is now correctly implemented according to the official AQ-10 autism screening questionnaire!** ✅

The app will now:
1. Present exact AQ-10 questions
2. Score them correctly (mixed forward/reverse)
3. Calculate accurate risk predictions
4. Match the trained model's expectations perfectly
