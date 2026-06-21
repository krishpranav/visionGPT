# VisionGPT 10B

## Dataset Architecture Specification

Version: 0.1

Status: Draft

---

# Table of Contents

* 1. Philosophy
* 2. Dataset Categories
* 3. Dataset Hierarchy
* 4. Perception Dataset
* 5. Scene Graph Dataset
* 6. Spatial Reasoning Dataset
* 7. Visual Reasoning Dataset
* 8. Synthetic Dataset Generation
* 9. Dataset Validation
* 10. Data Storage Format
* 11. Training Splits
* 12. Quality Standards

---

# 1. Philosophy

VisionGPT is a reasoning-first visual model.

Therefore the datasets must teach:

* Seeing
* Understanding
* Relationships
* Reasoning

Instead of only:

* Caption generation
* Visual description
* OCR

---

# Dataset Priority

```text id="0lye48"
Perception
 ↓
Relationships
 ↓
Spatial Facts
 ↓
Reasoning
 ↓
Language
```

Language is the final stage.

Reasoning is the core stage.

---

# 2. Dataset Categories

VisionGPT training data is divided into four major groups.

```text id="z5n4hx"
Group A
Perception

Group B
Scene Graph

Group C
Spatial Reasoning

Group D
Visual Reasoning
```

---

# 3. Dataset Hierarchy

```text id="9mgn5e"
datasets/

├── perception/
│
├── scene_graph/
│
├── spatial_reasoning/
│
├── visual_reasoning/
│
├── synthetic/
│
├── validation/
│
└── metadata/
```

---

# 4. Perception Dataset

## Purpose

Teach the model to recognize visual entities.

---

## Example

Input:

```text id="9gn0uo"
Image
```

Output:

```json id="ktc9lp"
{
  "objects": [
    {
      "label": "person",
      "bbox": [100, 150, 300, 700]
    }
  ]
}
```

---

## Target Classes

Initial V1:

```text id="gmlzvn"
People

Animals

Vehicles

Furniture

Buildings

Tools

Electronics

Food

Plants

Signs
```

---

## Attributes

Each object should support:

```json id="wpqv6d"
{
  "color": "",
  "shape": "",
  "material": "",
  "size": ""
}
```

---

## Target Size

```yaml id="o6ybws"
images:
  100M+

objects:
  3B+
```

---

# 5. Scene Graph Dataset

## Purpose

Teach relationships between entities.

---

## Example

Objects:

```json id="gk9hae"
{
  "person": 1,
  "umbrella": 1
}
```

Graph:

```json id="5ddn7n"
{
  "subject": "person",
  "relation": "holding",
  "object": "umbrella"
}
```

---

## Relationship Types

### Directional

```text id="1bngjf"
left_of

right_of

above

below
```

---

### Interaction

```text id="4c8rff"
holding

touching

using

carrying

wearing
```

---

### Spatial

```text id="95mb4r"
inside

outside

overlapping

behind

in_front_of
```

---

## Target Size

```yaml id="y0s37f"
relationships:
  500M+
```

---

# 6. Spatial Reasoning Dataset

## Purpose

Teach geometry and layout understanding.

---

## Example

Question:

```text id="o1d6j3"
How many chairs are present?
```

Answer:

```json id="6e4mtf"
{
  "count": 7
}
```

---

## Categories

### Counting

```text id="9hhplx"
1-100 objects
```

---

### Position

```text id="pq9pt0"
left

right

top

bottom
```

---

### Distance

```text id="4kws6g"
nearest

farthest
```

---

### Groups

```text id="h1q0x3"
clusters

rows

columns
```

---

### Paths

```text id="f7k7gm"
navigation

route finding
```

---

## Target Size

```yaml id="7e59vx"
samples:
  250M+
```

---

# 7. Visual Reasoning Dataset

## Purpose

Teach multi-step visual logic.

---

## Example

Question:

```text id="ixiwzh"
How many people are holding umbrellas?
```

Reasoning:

```text id="tjmqj3"
Step 1:
Find people

Step 2:
Find umbrellas

Step 3:
Find holding relationships

Step 4:
Count matches

Answer:
2
```

---

## Dataset Structure

```json id="4m4cmf"
{
  "question": "",
  "reasoning": [],
  "answer": ""
}
```

---

## Categories

### Counting Reasoning

```text id="0byzv5"
Count objects with constraints
```

Example:

```text id="v1o2g0"
How many red cars?
```

---

### Spatial Reasoning

```text id="yydwjo"
Find spatial relationships
```

Example:

```text id="p4og94"
What is left of the bus?
```

---

### Relational Reasoning

```text id="vs5i7l"
Object interactions
```

Example:

```text id="x8k7yw"
Who is holding the umbrella?
```

---

### Multi-Step Reasoning

Example:

```text id="9hrc6l"
Which person is standing behind the
person holding the backpack?
```

---

## Target Size

```yaml id="uq2xjx"
samples:
  100M+
```

---

# 8. Synthetic Dataset Generation

## Why Synthetic Data

Real datasets rarely contain reasoning traces.

VisionGPT requires reasoning supervision.

---

## Pipeline

```text id="2w85bh"
Scene Generator
 ↓
Object Placement
 ↓
Relationship Generation
 ↓
Reasoning Generation
 ↓
Validation
```

---

## Example

Generated Scene:

```json id="6sv4uk"
{
  "person": 3,
  "umbrella": 2
}
```

Generated Question:

```text id="6v9jd4"
How many umbrellas are being held?
```

Generated Reasoning:

```text id="1pjg9i"
Person A holds Umbrella 1

Person B holds Umbrella 2

Answer:
2
```

---

# 9. Dataset Validation

Every sample must pass:

---

## Structural Validation

```text id="r3h6ul"
Valid JSON

Valid Labels

Valid Relationships
```

---

## Logical Validation

```text id="3oq53t"
Reasoning must produce answer
```

---

## Spatial Validation

```text id="h4i0qb"
Bounding boxes must be valid
```

---

## Confidence Validation

```text id="i7e8mb"
Annotation confidence > threshold
```

---

# 10. Data Storage Format

## Images

```text id="0c5bqo"
WebP
```

---

## Metadata

```text id="dw6f2m"
JSON
```

---

## Shards

```text id="o6pqdc"
Parquet
```

---

## Example

```json id="okteo6"
{
  "image_id": "img_001",
  "objects": [],
  "relationships": [],
  "questions": [],
  "answers": []
}
```

---

# 11. Training Splits

```yaml id="1c34ua"
train: 90%

validation: 5%

test: 5%
```

---

## Hard Test Set

Must never appear during training.

Purpose:

```text id="5mqz1q"
Measure real reasoning ability.
```

---

# 12. Quality Standards

Every dataset entering training must satisfy:

---

## Object Accuracy

```yaml id="9ud47r"
minimum: 98%
```

---

## Relationship Accuracy

```yaml id="4vgpvn"
minimum: 95%
```

---

## Reasoning Accuracy

```yaml id="ej5b0w"
minimum: 95%
```

---

## Duplicate Threshold

```yaml id="e4xn18"
maximum: 3%
```

---

## Corrupted Data

```yaml id="n2u6hf"
maximum: 0.5%
```

---

# Dataset Principle

VisionGPT is trained on facts before language.

```text id="sxn38g"
Objects
 ↓
Relationships
 ↓
Spatial Facts
 ↓
Reasoning
 ↓
Language
```

If the facts are wrong, the reasoning is wrong.

If the reasoning is wrong, the answer is wrong.

Therefore data quality is the highest priority component of the entire system.
