# BEAVR Bench Scenes

BEAVR Bench includes several pre-built scenes designed to test different aspects of physical AI, from basic manipulation to advanced memory and precision.

## Core Concepts

Each scene is a [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) compliant environment with its own set of **Rules**. These rules handle:

- **State management** (e.g., animations, timers)
- **Success/Failure conditions**
- **Reward calculation**
- **Scene randomization**

---

## Pick and Place

A fundamental manipulation task where the robot must move an object to a specific target zone.

### Pick and Place Overview

The robot starts with a block at a randomized position on the table. It must grasp the block and place it precisely within a target goal site.

### ⚖️ Pick and Place Rules

- **Success**: The distance between the block and the goal site is below the success threshold.
- **Failure**: The block falls off the table or is dropped below a certain height.
- **Randomization**: The block's initial position on the table is randomized at the start of each episode.

---

## Shell Game

An advanced tracking and memory task designed to test object permanence.

### Shell Game Overview

The robot is shown a ball inside one of three identical cups. The cups are then covered and shuffled in a random sequence. The robot must lift the correct cup after the shuffle is finished.

### ⚖️ Shell Game Rules

- **Phase 1: Showing**: The target cup is lifted for 3 seconds to reveal the ball.
- **Phase 2: Covering**: The cup is lowered to conceal the ball.
- **Phase 3: Shuffling**: The cups are swapped in a randomized sequence.
- **Phase 4: Testing**: The robot must lift the cup that contains the ball.
- **Success**: The robot lifts the correct cup.
- **Failure**: The robot lifts an incorrect cup.

---

## Server Swap

A high-precision assembly task with a transient memory cue.

### Server Swap Overview

A mobile manipulator is tasked with replacing a faulty server drive. The faulty drive is identified by a transient orange LED signal that vanishes after 5 seconds.

### ⚖️ Server Swap Rules

- **Phase 1: Cue**: One failing slot shows an orange LED; others are green.
- **Phase 2: Vanished**: All LEDs turn green, requiring the robot to have memorized the position.
- **Success**: The replacement drive is inserted into the correct (memorized) slot.
- **Failure**: The drive is placed in the wrong slot or dropped.
- **Randomization**: The failing slot index is randomized per episode.

---

## Vanishing Blueprint

A sequential memory and stacking task.

### Vanishing Blueprint Overview

The robot must replicate a specific stack of colored blocks shown as a holographic blueprint. The blueprint vanishes after 5 seconds, requiring the robot to recall the exact order.

### ⚖️ Vanishing Blueprint Rules

- **Phase 1: Showing**: Holograms show the desired stack order (bottom to top).
- **Phase 2: Testing**: The holograms vanish, and the robot must stack physical blocks.
- **Success**: Blocks are stacked in the build zone in the exact sequence specified by the blueprint.
- **Randomization**: The stack order is randomized at the start of each episode.
