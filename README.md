# :bookmark: Awesome-VLA-Post-Training

**Awesome-VLA-Post-Training** is a continuously updated collection of cutting-edge resources focused on the **post-training of VLA systems**. As embodied AI experiences rapid growth, this repository serves as a centralized hub for research updates, practical codes, and implementation insights. Our goal is to enhance the ability of VLA agents to perceive, reason, and act within physical environments. Key focus areas include:

* :earth_asia: **Enhancing environmental perception**
* :brain: **Improving embodiment awareness**
* :memo: **Deepening task comprehension and generalization**
* :wrench: **Integrating and tuning multiple components**

We welcome contributions from researchers and practitioners passionate about advancing VLA systems. Join us in building a structured, high-quality resource for the community!

* **[2025-6]** :fire: Our paper, “Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends,” is now publicly available. ([Paper](https://arxiv.org/abs/2506.20966))

---

## :star: Notable Works

This is a curated selection of influential papers, benchmarks and projects that have made a significant contribution to the field of VLA systems. These works provide foundational insights and state-of-the-art methods that inform current research directions.

* **[2022-12]** RT-1: Robotics Transformer for real-world control at scale. ([Paper](https://arxiv.org/abs/2212.06817), [Website](https://robotics-transformer1.github.io), [Code](https://github.com/google-research/robotics_transformer))

* **[2023-07]** RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. ([Paper](https://arxiv.org/abs/2307.15818), [Website](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action))

* **[2023-10]** Octopus: Embodied Vision-Language Programmer from Environmental Feedback. ([Paper](https://arxiv.org/abs/2310.08588), [Website](https://choiszt.github.io/Octopus), [Code](https://github.com/dongyh20/Octopus))

* **[2023-11]** Vision-Language Foundation Models as Effective Robot Imitators. ([Paper](https://arxiv.org/abs/2311.01378), [Website](https://roboflamingo.github.io), [Code](https://github.com/RoboFlamingo/RoboFlamingo))

* **[2023-12]** Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation. ([Paper](https://arxiv.org/abs/2312.13139), [Website](https://gr1-manipulation.github.io), [Code](https://github.com/GR1-Manipulation/GR-1))

* **[2024-03]** 3D-VLA: A 3D Vision-Language-Action Generative World Model. ([Paper](https://arxiv.org/abs/2403.09631), [Website](https://vis-www.cs.umass.edu/3dvla), [Code](https://github.com/UMass-Embodied-AGI/3D-VLA))

* **[2024-05]** Octo: An Open-Source Generalist Robot Policy. ([Paper](https://arxiv.org/abs/2405.12213), [Website](https://octo-models.github.io), [Code](https://github.com/octo-models/octo))

* **[2024-06]** OpenVLA: An Open-Source Vision-Language-Action Model. ([Paper](https://arxiv.org/abs/2406.09246), [Website](https://openvla.github.io), [Code](https://github.com/openvla/openvla))

* **[2024-06]** RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation. ([Paper](https://arxiv.org/abs/2406.04339), [Website](https://sites.google.com/view/robomamba-web), [Code](https://github.com/lmzpai/roboMamba))

* **[2024-10]** RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation. ([Paper](https://arxiv.org/abs/2410.07864), [Website](https://rdt-robotics.github.io/rdt-robotics), [Code](https://github.com/thu-ml/RoboticsDiffusionTransformer))

* **[2024-10]** π0: A Vision-Language-Action Flow Model for General Robot Control. ([Paper](https://arxiv.org/abs/2410.24164), [Website](https://www.physicalintelligence.company/blog/pi0), [Code](https://github.com/Physical-Intelligence/openpi))

* **[2024-11]** CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation. ([Paper](https://arxiv.org/abs/2411.19650), [Website](https://cogact.github.io), [Code](https://github.com/microsoft/CogACT))

* **[2025-04]** π0.5: a Vision-Language-Action Model with Open-World Generalization. ([Paper](https://arxiv.org/abs/2504.16054), [Website](https://www.physicalintelligence.company/blog/pi05))

* **[2025-05]** UniVLA: Learning to Act Anywhere with Task-centric Latent Actions. ([Paper](https://arxiv.org/abs/2505.06111), [Code](https://github.com/OpenDriveLab/UniVLA))

* **[2025-06]** SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. ([Paper](https://arxiv.org/abs/2506.01844), [Code](https://github.com/huggingface/lerobot))

---

## :earth_asia: Enhancing Environmental Perception

This section explores methods that improve an agent’s ability to perceive and interpret its environment. It includes **affordance-guided learning**, which enables agents to understand actionable properties of objects; **enhanced encoders** tailored for manipulation tasks, allowing more precise feature extraction; and **improved representation learning**, which helps models build richer and more structured environmental understanding for downstream tasks.

### Affordance-Guided Learning

* **[2024-02]** RoboCodeX: Multimodal Code Generation for Robotic Behavior Synthesis. ([Paper](https://arxiv.org/abs/2402.16117), [Website](https://sgtvincent.github.io/publication/mu-2024-robocodex))

* **[2024-03]** RT-H: Action Hierarchies Using Language. ([Paper](https://arxiv.org/abs/2403.01823), [Website](https://rt-hierarchy.github.io/))

* **[2024-06]** A3VLM: Actionable Articulation-Aware Vision Language Model. ([Paper](https://arxiv.org/abs/2406.07549), [Code](https://github.com/changhaonan/A3VLM))

* **[2024-06]** RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics. ([Paper](https://arxiv.org/abs/2406.10721), [Website](https://robo-point.github.io), [Code](https://github.com/wentaoyuan/RoboPoint))

* **[2024-11]** RT-Affordance: Affordances are Versatile Intermediate Representations for Robot Manipulation. ([Paper](https://arxiv.org/abs/2411.02704), [Website](https://snasiriany.me/rt-affordance))

* **[2024-12]** Improving Vision-Language-Action Models via Chain-of-Affordance. ([Paper](https://arxiv.org/abs/2412.20451))

* **[2025-01]** OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints. ([Paper](https://arxiv.org/abs/2501.03841), [Website](https://omnimanip.github.io), [Code](https://github.com/pmj110119/OmniManip))

* **[2025-04]** RoboAct-CLIP: Video-Driven Pre-training of Atomic Action Understanding for Robotics. ([Paper](https://arxiv.org/abs/2504.02069))

* **[2025-04]** A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation. ([Paper](https://arxiv.org/abs/2504.12636))

* **[2025-04]** ControlManip: Few-Shot Manipulation Fine-tuning via Object-centric Conditional Control. ([Paper](https://openreview.net/forum?id=3LvTtj0VYy))

### Enhanced Encoder for Manipulation

* **[2024-02]** Task-conditioned adaptation of visual features in multi-task policy learning. ([Paper](https://arxiv.org/abs/2402.07739), [Website](https://pierremarza.github.io/projects/task_conditioned_adaptation), [Code](https://github.com/PierreMarza/task_conditioned_adaptation))

* **[2024-03]** Never-Ending Behavior-Cloning Agent for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2403.00336), [Website](https://neragent.github.io))

* **[2024-07]** Theia: Distilling Diverse Vision Foundation Models for Robot Learning. ([Paper](https://arxiv.org/abs/2407.20179), [Website](https://theia.theaiinstitute.com), [Code](https://github.com/bdaiinstitute/theia))

* **[2024-09]** TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2409.12514), [Website](https://tiny-vla.github.io), [Code](https://github.com/JayceWen/tinyvla))

* **[2024-10]** M2Distill: Multi-Modal Distillation for Lifelong Imitation Learning. ([Paper](https://arxiv.org/abs/2410.00064))

* **[2024-11]** RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics. ([Paper](https://arxiv.org/abs/2411.16537), [Website](https://chanh.ee/RoboSpatial), [Code](https://github.com/NVlabs/RoboSpatial))

* **[2025-02]** ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model. ([Paper](https://arxiv.org/abs/2502.14420))

* **[2025-03]** MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation. ([Paper](https://arxiv.org/abs/2503.20384))

* **[2025-03]** A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning. ([Paper](https://arxiv.org/abs/2503.06960), [Code](https://github.com/CVMI-Lab/SlotMIM))

### Enhanced Representation for Manipulation

* **[2024-02]** Vision-Language Models Provide Promptable Representations for Reinforcement Learning. ([Paper](https://arxiv.org/abs/2402.02651), [Website](https://pr2l.github.io), [Code](https://github.com/pr2l/pr2l.github.io/blob/master/static/notebooks/PR2LExample.ipynb))

* **[2024-03]** Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics. ([Paper](https://arxiv.org/abs/2403.19578))

* **[2024-05]** Pre-trained Text-to-Image Diffusion Models Are Versatile Representation Learners for Control. ([Paper](https://arxiv.org/abs/2405.05852))

* **[2025-01]** SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model. ([Paper](https://arxiv.org/abs/2501.15830), [Website](https://spatialvla.github.io), [Code](https://github.com/SpatialVLA/SpatialVLA))

* **[2025-02]** BFA: Best-Feature-Aware Fusion for Multi-View Fine-grained Manipulation. ([Paper](https://arxiv.org/abs/2502.11161))

* **[2025-02]** VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation. ([Paper](https://arxiv.org/abs/2502.02175), [Website](https://vla-cache.github.io), [Code](https://github.com/siyuhsu/vla-cache))

* **[2025-02]** VLAS: Vision-Language-Action Model With Speech Instructions For Customized Robot Manipulation. ([Paper](https://arxiv.org/abs/2502.13508))

* **[2025-02]** ObjectVLA: End-to-End Open-World Object Manipulation Without Demonstration. ([Paper](https://arxiv.org/abs/2502.19250), [Website](https://objectvla.github.io))

* **[2025-02]** DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping. ([Paper](https://arxiv.org/abs/2502.20900), [Website](https://dexgraspvla.github.io), [Code](https://github.com/Psi-Robot/DexGraspVLA))

* **[2025-03]** OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction. ([Paper](https://arxiv.org/abs/2503.03734), [Website](https://ottervla.github.io), [Code](https://github.com/Max-Fu/otter))

* **[2025-05]** VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation. ([Paper](https://arxiv.org/abs/2505.09577), [Website](https://sites.google.com/view/vtla))

* **[2025-06]** BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models. ([Paper](https://www.arxiv.org/abs/2506.07961), [Website](https://bridgevla.github.io), [Code](https://github.com/BridgeVLA/BridgeVLA))

---

## :brain: Improving Embodiment Awareness

Here we focus on helping agents better understand their own physical structure and capabilities. Topics include **forward and inverse kinematics learning**, which allow agents to model the relationship between joint movements and spatial positions, and **action head design**, aimed at optimizing how high-level decisions are translated into low-level motor commands.

### Forward and Inverse kinematics learning

* **[2023-10]** Mastering robot manipulation with multimodal prompts through pretraining and multi-task fine-tuning. ([Paper](https://arxiv.org/abs/2310.09676), [Website](https://midas-icml.github.io), [Code]())

* **[2024-10]** Effective Tuning Strategies for Generalist Robot Manipulation Policies. ([Paper](https://arxiv.org/abs/2410.01220))

* **[2024-12]** Learning Novel Skills from Language-Generated Demonstrations. ([Paper](https://arxiv.org/abs/2412.09286), [Website](https://aoqunjin.github.io/LNSLGD), [Code](https://github.com/AoqunJin/LNSLGD))

* **[2025-02]** HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation. ([Paper](https://arxiv.org/abs/2502.05485), [Website](https://hamster-robot.github.io), [Code](https://github.com/liyi14/HAMSTER_beta))

* **[2025-05]** LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning. ([Paper](https://arxiv.org/abs/2406.11815), [Website](https://llarva24.github.io), [Code](https://github.com/Dantong88/LLARVA))

### Action Head Designing

* **[2023-10]** TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models. ([Paper](https://arxiv.org/pdf/2310.05905))

* **[2024-05]** FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies. ([Paper](https://openreview.net/forum?id=ifo8oWSLSq))

* **[2024-06]** Grounding Multimodal Large Language Models in Actions. ([Paper](https://arxiv.org/abs/2406.07904))

* **[2024-08]** Bidirectional Decoding: Improving Action Chunking via Closed-Loop Resampling. ([Paper](https://arxiv.org/abs/2408.17355), [Website](https://bid-robot.github.io), [Code](https://github.com/YuejiangLIU/bid_diffusion))

* **[2024-09]** Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers. ([Paper](https://arxiv.org/pdf/2409.20537))

* **[2024-10]** Vision-Language-Action Model and Diffusion Policy Switching Enables Dexterous Control of an Anthropomorphic Hand. ([Paper](https://arxiv.org/abs/2410.14022))

* **[2025-01]** FAST: Efficient Action Tokenization for Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2501.09747), [Website](https://www.physicalintelligence.company/research/fast))

* **[2025-01]** Universal Actions for Enhanced Embodied Foundation Models. ([Paper](https://arxiv.org/pdf/2501.10105), [Website](https://2toinf.github.io/UniAct/?utm_source=tldrai), [Code](https://github.com/2toinf/UniAct))

* **[2025-02]** Fine-tuning vision-language-action models: Optimizing speed and success. ([Paper](https://arxiv.org/abs/2502.19645), [Website](https://openvla-oft.github.io), [Code](https://github.com/moojink/openvla-oft))

* **[2025-03]** Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding. ([Paper](https://arxiv.org/abs/2503.02310))

* **[2025-03]** Refined Policy Distillation: From VLA Generalists to RL Experts. ([Paper](https://arxiv.org/abs/2503.05833))

* **[2025-03]** HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model. ([Paper](https://arxiv.org/abs/2503.10631), [Website](https://hybrid-vla.github.io), [Code](https://github.com/PKU-HMI-Lab/Hybrid-VLA))

* **[2025-03]** Efficient Continual Adaptation of Pretrained Robotic Policy with Online Meta-Learned Adapters. ([Paper](https://arxiv.org/pdf/2503.18684))

* **[2025-03]** Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy. ([Paper](https://arxiv.org/abs/2503.19757), [Website](https://robodita.github.io), [Code](https://github.com/RoboDita/Dita))

---

## :memo: Deepening Task Comprehension

This section covers methods that enable agents to better understand and generalize across tasks. Key areas include **human–robot interaction**, where agents learn to interpret and respond to human inputs effectively, and **hierarchical task manipulation**, which enables multi-step reasoning and planning by decomposing complex tasks into structured subtasks.

### Human–Robot-Interaction

* **[2023-10]** What Matters to You? Towards Visual Representation Alignment for Robot Learning. ([Paper](https://arxiv.org/abs/2310.07932))

* **[2024-05]** Hummer: Towards Limited Competitive Preference Dataset. ([Paper](https://arxiv.org/abs/2405.11647))

* **[2024-12]** Maximizing Alignment with Minimal Feedback: Efficiently Learning Rewards for Visuomotor Robot Policy Alignment. ([Paper](https://arxiv.org/abs/2412.04835))

* **[2025-03]** Adversarial Data Collection: Human-Collaborative Perturbations for Efficient and Robust Robotic Imitation Learning. ([Paper](https://arxiv.org/abs/2503.11646))

* **[2025-03]** VLA Model-Expert Collaboration for Bi-directional Manipulation Learning. ([Paper](https://arxiv.org/abs/2503.04163), [Website](https://aoqunjin.github.io/Expert-VLA))

* **[2025-03]** RoboCopilot: Human-in-the-loop Interactive Imitation Learning for Robot Manipulation. ([Paper](https://arxiv.org/abs/2503.07771))

### Hierarchical Task Manipulation

* **[2023-11]** Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning. ([Paper](https://arxiv.org/abs/2311.17842), [Website](https://robot-vila.github.io), [Code]())

* **[2024-07]** Diffusion Augmented Agents: A Framework for Efficient Exploration and Transfer Learning. ([Paper](https://arxiv.org/abs/2407.20798))

* **[2024-08]** Policy Adaptation via Language Optimization: Decomposing Tasks for Few-Shot Imitation. ([Paper](https://arxiv.org/abs/2408.16228), [Website](https://palo-website.github.io), [Code](https://github.com/vivekmyers/palo))

* **[2024-11]** STEER: Flexible Robotic Manipulation via Dense Language Grounding. ([Paper](https://arxiv.org/abs/2411.03409), [Website](https://lauramsmith.github.io/steer))

* **[2024-11]** GRAPE: Generalizing Robot Policy via Preference Alignment. ([Paper](https://arxiv.org/abs/2411.19309), [Website](https://grape-vla.github.io), [Code](https://github.com/aiming-lab/GRAPE))

* **[2024-11]** CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision. ([Paper](https://arxiv.org/abs/2411.00508), [Website](https://clip-rt.github.io), [Code](https://github.com/gicheonkang/clip-rt))

* **[2025-02]** RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete. ([Paper](https://arxiv.org/abs/2502.21257), [Website](https://superrobobrain.github.io), [Code](https://github.com/FlagOpen/RoboBrain))

* **[2025-03]** RoboDexVLM: Visual Language Model-Enabled Task Planning and Motion Control for Dexterous Robot Manipulation. ([Paper](https://arxiv.org/abs/2503.01616), [Website](https://henryhcliu.github.io/robodexvlm), [Code]())

* **[2025-03]** DataPlatter: Boosting Robotic Manipulation Generalization with Minimal Costly Data. ([Paper](https://arxiv.org/abs/2503.19516))

* **[2025-05]** LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning. ([Paper](https://arxiv.org/abs/2406.11815), [Website](https://llarva24.github.io), [Code](https://github.com/Dantong88/LLARVA))

* **[2025-05]** OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning. ([Paper](https://arxiv.org/abs/2505.11917), [Website](https://one-two-vla.github.io), [Code](https://github.com/Fanqi-Lin/OneTwoVLA))

* **[2025-05]** Pre-Trained Multi-Goal Transformers with Prompt Optimization for Efficient Online Adaptation. ([Paper](https://openreview.net/forum?id=DHucngOEe3&noteId=w5HGYk50VI))

---

## :wrench: Multiple Component Integration

Integrating various subsystems is essential for building robust VLA agents. This section includes **reinforcement learning frameworks** for continuous control and decision-making, **visual interaction prediction** for anticipating future outcomes based on perception, and strategies for **active dataset processing** to reduce the cost of adapting models to new environments or tasks.

### Reinforcement Learning

* **[2023-10]** Unleashing the Power of Pre-trained Language Models for Offline Reinforcement Learning. ([Paper](https://arxiv.org/abs/2310.20587), [Website](https://lamo2023.github.io), [Code](https://github.com/srzer/LaMo-2023))

* **[2023-12]** LiFT: Unsupervised Reinforcement Learning with Foundation Models as Teachers. ([Paper](https://arxiv.org/abs/2312.08958))

* **[2024-01]** Building Open-Ended Embodied Agent via Language-Policy Bidirectional Adaptation. ([Paper](https://arxiv.org/abs/2401.00006), [Code](https://github.com/opendilab/OpenPaL))

* **[2024-01]** Improving Vision-Language-Action Model with Online Reinforcement Learning. ([Paper](https://arxiv.org/abs/2501.16664))

* **[2024-01]** Vintix: Action Model via In-Context Reinforcement Learning. ([Paper](https://arxiv.org/abs/2501.19400), [Website](), [Code](https://github.com/dunnolab/vintix))

* **[2024-02]** ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy. ([Paper](https://arxiv.org/abs/2502.05450), [Website](https://cccedric.github.io/conrft), [Code](https://github.com/cccedric/conrft))

* **[2024-02]** A Real-to-Sim-to-Real Approach to Robotic Manipulation with VLM-Generated Iterative Keypoint Rewards. ([Paper](https://arxiv.org/pdf/2502.08643), [Website](https://iker-robot.github.io), [Code](https://github.com/shivanshpatel35/IKER))

* **[2024-02]** Learning a High-quality Robotic Wiping Policy Using Systematic Reward Analysis and Visual-Language Model Based Curriculum. ([Paper](https://arxiv.org/abs/2502.12599))

* **[2024-02]** Offline Actor-Critic Reinforcement Learning Scales to Large Models. ([Paper](https://arxiv.org/abs/2402.05546))

* **[2024-05]** PEAC: Unsupervised Pre-training for Cross-Embodiment Reinforcement Learning. ([Paper](https://arxiv.org/abs/2405.14073), [Website](https://yingchengyang.github.io/ceurl), [Code](https://github.com/thu-ml/CEURL))

* **[2024-07]** Affordance-Guided Reinforcement Learning via Visual Prompting. ([Paper](https://arxiv.org/abs/2407.10341))

* **[2024-09]** FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning. ([Paper](https://arxiv.org/abs/2409.16578), [Website](https://robot-flare.github.io), [Code](https://github.com/JiahengHu/FLaRe))

* **[2024-09]** Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving. ([Paper](https://arxiv.org/abs/2409.18343))

* **[2024-10]** Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance. ([Paper](https://arxiv.org/abs/2410.13816), [Website](https://nakamotoo.github.io/V-GPS), [Code](https://github.com/nakamotoo/V-GPS))

* **[2024-10]** GRAPPA: Generalizing and Adapting Robot Policies via Online Agentic Guidance. ([Paper](https://arxiv.org/abs/2410.06473))

* **[2024-12]** Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Class and Backbone. ([Paper](https://arxiv.org/abs/2412.06685), [Website](https://policyagnosticrl.github.io), [Code](https://github.com/MaxSobolMark/PolicyAgnosticRL))

* **[2025-01]** FDPP: Fine-tune Diffusion Policy with Human Preference. ([Paper](https://arxiv.org/abs/2501.08259))

* **[2025-05]** Lifelong Autonomous Improvement of Navigation Foundation Models in the Wild. ([Paper](https://openreview.net/forum?id=vBj5oC60Lk), [Code](https://github.com/kylestach/lifelong-nav-rl))

* **[2025-05]** What Can RL Bring to VLA Generalization? An Empirical Study. ([Paper](https://arxiv.org/abs/2505.19789), [Website](https://rlvla.github.io), [Code](https://github.com/gen-robot/RL4VLA))

* **[2025-05]** VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning. ([Paper](https://arxiv.org/abs/2505.18719), [Code](https://github.com/GuanxingLu/vlarl))

* **[2025-05]** Interactive Post-Training for Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2505.17016), [Website](https://ariostgx.github.io/ript_vla), [Code](https://github.com/Ariostgx/ript-vla))

### Visual Interaction Prediction

* **[2023-12]** Unleashing large-scale video generative pre-training for visual robot manipulation. ([Paper](https://arxiv.org/abs/2312.13139), [Website](https://gr1-manipulation.github.io), [Code](https://github.com/bytedance/GR-1))

* **[2024-03]** MineDreamer: Learning to Follow Instructions via Chain-of-Imagination for Simulated-World Control. ([Paper](https://arxiv.org/abs/2403.12037), [Website](https://sites.google.com/view/minedreamer/main), [Code](https://github.com/Zhoues/MineDreamer))

* **[2024-06]** Learning Manipulation by Predicting Interaction. ([Paper](https://arxiv.org/abs/2406.00439), [Website](https://opendrivelab.com/MPI), [Code](https://github.com/OpenDriveLab/MPI))

* **[2024-06]** Language-Guided Manipulation with Diffusion Policies and Constrained Inpainting. ([Paper](https://arxiv.org/abs/2406.09767), [Website](https://disco2025.github.io))

* **[2024-07]** VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2407.09829), [Code](https://github.com/ppjmchen/vlmpc))

* **[2024-07]** Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals. ([Paper](https://arxiv.org/abs/2407.05996), [Website](https://intuitive-robots.github.io/mdt_policy), [Code](https://github.com/intuitive-robots/mdt_policy))

* **[2024-08]** GR-MG: Leveraging Partially-Annotated Data Via Multi-Modal Goal-Conditioned Policy. ([Paper](https://arxiv.org/abs/2408.14368), [Website](https://gr-mg.github.io), [Code](https://github.com/bytedance/GR-MG/tree/main))

* **[2024-10]** Run-time Observation Interventions Make Vision-Language-Action Models More Visually Robust. ([Paper](https://arxiv.org/abs/2410.01971), [Website](https://aasherh.github.io/byovla), [Code](https://github.com/irom-princeton/byovla))

* **[2024-10]** VIP: Vision Instructed Pre-training for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2410.07169), [Website](https://lizhuoling.github.io/VIRT_webpage), [Code](https://github.com/Lizhuoling/VIRT))

* **[2024-10]** Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. ([Paper](https://arxiv.org/abs/2410.06158), [Website](https://gr2-manipulation.github.io))

* **[2024-12]** Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations. ([Paper](https://arxiv.org/abs/2412.14803), [Website](https://github.com/roboterax/video-prediction-policy), [Code](https://video-prediction-policy.github.io))

* **[2024-12]** Moto: Latent motion token as the bridging language for robot manipulation. ([Paper](https://arxiv.org/abs/2412.04445), [Website](https://chenyi99.github.io/moto), [Code](https://github.com/TencentARC/Moto))

* **[2024-12]** Predictive inverse dynamics models are scalable learners for robotic manipulation. ([Paper](https://arxiv.org/abs/2412.15109), [Website](https://nimolty.github.io/Seer), [Code](https://github.com/OpenRobotLab/Seer))

* **[2025-01]** UP-VLA: A Unified Understanding and Prediction Model for Embodied Agent. ([Paper](https://arxiv.org/abs/2501.18867))

* **[2025-02]** HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation. ([Paper](https://arxiv.org/abs/2502.05485), [Website](https://hamster-robot.github.io), [Code](https://github.com/liyi14/HAMSTER_beta))

* **[2025-03]** Unified Video Action Model. ([Paper](https://arxiv.org/abs/2503.00200), [Website](https://unified-video-action-model.github.io), [Code](https://github.com/ShuangLI59/unified_video_action))

* **[2025-04]** Vision-Language Model Predictive Control for Manipulation Planning and Trajectory Generation. ([Paper](https://arxiv.org/abs/2504.05225), [Code](https://github.com/ppjmchen/vlmpc))

### Active Dataset Processing

* **[2023-11]** RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation. ([Paper](https://arxiv.org/abs/2311.01455), [Website](https://robogen-ai.github.io), [Code](https://github.com/Genesis-Embodied-AI/RoboGen))

* **[2024-01]** SWBT: Similarity Weighted Behavior Transformer with the Imperfect Demonstration for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2401.08957))

* **[2024-02]** Transductive Active Learning: Theory and Applications. ([Paper](https://arxiv.org/abs/2402.15898), [Code](https://github.com/jonhue/transductive-active-learning))

* **[2024-03]** Efficient Data Collection for Robotic Manipulation via Compositional Generalization. ([Paper](https://arxiv.org/abs/2403.05110))

* **[2024-06]** RVT-2: Learning Precise Manipulation from Few Demonstrations. ([Paper](https://arxiv.org/abs/2406.08545), [Website](https://robotic-view-transformer-2.github.io), [Code](https://github.com/nvlabs/rvt))

* **[2024-07]** Autonomous Improvement of Instruction Following Skills via Foundation Models. ([Paper](https://arxiv.org/abs/2407.20635), [Website](https://auto-improvement.github.io), [Code](https://github.com/rail-berkeley/soar))

* **[2024-10]** Active Fine-Tuning of Generalist Policies. ([Paper](https://arxiv.org/abs/2410.05026))

* **[2024-10]** Data Scaling Laws in Imitation Learning for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2410.18647), [Website](https://data-scaling-laws.github.io), [Code](https://github.com/Fanqi-Lin/Data-Scaling-Laws))

* **[2025-02]** RoboBERT: An End-to-end Multimodal Robotic Manipulation Model. ([Paper](https://arxiv.org/abs/2502.07837))

* **[2025-02]** DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control. ([Paper](https://arxiv.org/abs/2502.05855), [Website](https://dex-vla.github.io), [Code](https://github.com/juruobenruo/DexVLA))

* **[2025-02]** DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning. ([Paper](https://arxiv.org/abs/2502.16932), [Website](https://demo-generation.github.io), [Code](https://github.com/TEA-Lab/DemoGen))

* **[2025-03]** DataPlatter: Boosting Robotic Manipulation Generalization with Minimal Costly Data. ([Paper](https://arxiv.org/abs/2503.19516))

---

## :black_nib: Contributing

We welcome contributions from the community! Whether it's adding new papers, sharing code, or improving documentation, your input helps make this a valuable resource for everyone!

---

## :pushpin: BibTeX

To cite this repository in your research, please use the following BibTeX entry:

```bibtex
@misc{awesome_embodied_ai_tuning,
  title       = {Awesome-VLA-Post-Training},
  author      = {Contributors},
  year        = {2025},
  howpublished= {\url{https://github.com/AoqunJin/Awesome-VLA-Post-Training}},
  note        = {A collection of VLA post-training methods}
}
```
