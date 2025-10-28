# :bookmark: Awesome-VLA-Post-Training

**Awesome-VLA-Post-Training** is a continuously updated collection of cutting-edge resources focused on the **post-training of VLA systems**. As embodied AI experiences rapid growth, this repository serves as a centralized hub for research updates, practical codes, and implementation insights. Our goal is to enhance the ability of VLA agents to perceive, reason, and act within physical environments. Key focus areas include:

* :earth_asia: **Enhancing environmental perception**
* :brain: **Improving embodiment awareness**
* :memo: **Deepening task comprehension and generalization**
* :wrench: **Integrating and tuning multiple components**

We welcome contributions from researchers and practitioners passionate about advancing VLA systems. Join us in building a structured, high-quality resource for the community!

* **[2025-6]** Our paper, “Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends,” is now publicly available. ([Paper](https://arxiv.org/abs/2506.20966))
* **[2025-10]** :fire: We updated to version v2, adding 50+ new papers and benchmarks for comparison.

---

## :star: Notable Works

This is a curated selection of influential papers, benchmarks and projects that have made a significant contribution to the field of VLA systems. These works provide foundational insights and state-of-the-art methods that inform current research directions.

* **[2022-12]** RT-1: Robotics Transformer for real-world control at scale. ([Paper](https://arxiv.org/abs/2212.06817), [Website](https://robotics-transformer1.github.io), [Code](https://github.com/google-research/robotics_transformer))

* **[2023-07]** RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. ([Paper](https://arxiv.org/abs/2307.15818), [Website](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action))

* **[2024-03]** 3D-VLA: A 3D Vision-Language-Action Generative World Model. ([Paper](https://arxiv.org/abs/2403.09631), [Website](https://vis-www.cs.umass.edu/3dvla), [Code](https://github.com/UMass-Embodied-AGI/3D-VLA))

* **[2024-05]** Octo: An Open-Source Generalist Robot Policy. ([Paper](https://arxiv.org/abs/2405.12213), [Website](https://octo-models.github.io), [Code](https://github.com/octo-models/octo))

* **[2024-06]** OpenVLA: An Open-Source Vision-Language-Action Model. ([Paper](https://arxiv.org/abs/2406.09246), [Website](https://openvla.github.io), [Code](https://github.com/openvla/openvla))

* **[2024-06]** RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation. ([Paper](https://arxiv.org/abs/2406.04339), [Website](https://sites.google.com/view/robomamba-web), [Code](https://github.com/lmzpai/roboMamba))

* **[2024-10]** RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation. ([Paper](https://arxiv.org/abs/2410.07864), [Website](https://rdt-robotics.github.io/rdt-robotics), [Code](https://github.com/thu-ml/RoboticsDiffusionTransformer))

* **[2024-10]** π0: A Vision-Language-Action Flow Model for General Robot Control. ([Paper](https://arxiv.org/abs/2410.24164), [Website](https://www.physicalintelligence.company/blog/pi0), [Code](https://github.com/Physical-Intelligence/openpi))

* **[2024-10]** GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation. ([Paper](https://arxiv.org/abs/2410.06158), [Website](https://gr2-manipulation.github.io))

* **[2024-11]** CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation. ([Paper](https://arxiv.org/abs/2411.19650), [Website](https://cogact.github.io), [Code](https://github.com/microsoft/CogACT))

* **[2025-03]** Gemini Robotics: Bringing AI into the Physical World. ([Paper](https://arxiv.org/abs/2503.20020), [Website](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world))

* **[2025-03]** GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. ([Paper](https://arxiv.org/abs/2503.14734), [Website](https://developer.nvidia.com/isaac/gr00t), [Code](https://github.com/NVIDIA/Isaac-GR00T))

* **[2025-03]** AgiBot World Colosseo: A Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems. ([Paper](https://arxiv.org/abs/2503.06669), [Website](https://agibot-world.com), [Code](https://github.com/OpenDriveLab/AgiBot-World))

* **[2025-04]** π0.5: a Vision-Language-Action Model with Open-World Generalization. ([Paper](https://arxiv.org/abs/2504.16054), [Website](https://www.physicalintelligence.company/blog/pi05))

* **[2025-05]** UniVLA: Learning to Act Anywhere with Task-centric Latent Actions. ([Paper](https://arxiv.org/abs/2505.06111), [Code](https://github.com/OpenDriveLab/UniVLA))

* **[2025-05]** GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data. ([Paper](https://arxiv.org/abs/2505.03233), [Website](https://pku-epic.github.io/GraspVLA-web), [Code](https://github.com/PKU-EPIC/GraspVLA))

* **[2025-06]** SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. ([Paper](https://arxiv.org/abs/2506.01844), [Code](https://github.com/huggingface/lerobot))

* **[2025-07]** GR-3 Technical Report. ([Paper](https://arxiv.org/abs/2507.15493), [Website](https://seed.bytedance.com/GR3))

---

## :triangular_ruler: Benchmark

LIBERO and Calvin, two widely used simulation environments, along with several validated Vision-Language-Action models, are summarized below.

### LIBERO

![LIBERO](assets/LIBERO.png)

### Calvin

![Calvin](assets/Calvin.png)

---

## :earth_asia: Enhancing Environmental Perception

This section explores methods that improve an agent’s ability to perceive and interpret its environment. It includes **affordance-guided learning**, which enables agents to understand actionable properties of objects; **enhanced encoders** tailored for manipulation tasks, allowing more precise feature extraction; and **improved representation learning**, which helps models build richer and more structured environmental understanding for downstream tasks.

### Affordance-Guided Learning

* **[2024-01]** Object-Centric Instruction Augmentation for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2401.02814))

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

* **[2025-07]** VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2506.17561))

* **[2025-07]** InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation. ([Paper](https://arxiv.org/abs/2507.17520), [Website](https://yangs03.github.io/InstructVLA_Home), [Code](https://github.com/InternRobotics/InstructVLA))

### Enhanced Encoder for Manipulation

* **[2024-02]** Task-conditioned adaptation of visual features in multi-task policy learning. ([Paper](https://arxiv.org/abs/2402.07739), [Website](https://pierremarza.github.io/projects/task_conditioned_adaptation), [Code](https://github.com/PierreMarza/task_conditioned_adaptation))

* **[2024-03]** Never-Ending Behavior-Cloning Agent for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2403.00336), [Website](https://neragent.github.io))

* **[2024-06]** Learning Efficient and Robust Language-conditioned Manipulation using Textual-Visual Relevancy and Equivariant Language Mapping. ([Paper](https://arxiv.org/abs/2406.15677), [Website](https://mingxi-jia.github.io/gem_page), [Code](https://github.com/SaulBatman/GEM_code))

* **[2024-07]** Theia: Distilling Diverse Vision Foundation Models for Robot Learning. ([Paper](https://arxiv.org/abs/2407.20179), [Website](https://theia.theaiinstitute.com), [Code](https://github.com/bdaiinstitute/theia))

* **[2024-09]** TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2409.12514), [Website](https://tiny-vla.github.io), [Code](https://github.com/JayceWen/tinyvla))

* **[2024-10]** M2Distill: Multi-Modal Distillation for Lifelong Imitation Learning. ([Paper](https://arxiv.org/abs/2410.00064))

* **[2024-10]** VIRT: Vision Instructed Transformer for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2410.07169v1))

* **[2024-11]** RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics. ([Paper](https://arxiv.org/abs/2411.16537), [Website](https://chanh.ee/RoboSpatial), [Code](https://github.com/NVlabs/RoboSpatial))

* **[2025-02]** ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model. ([Paper](https://arxiv.org/abs/2502.14420))

* **[2025-03]** MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation. ([Paper](https://arxiv.org/abs/2503.20384))

* **[2025-03]** A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning. ([Paper](https://arxiv.org/abs/2503.06960), [Code](https://github.com/CVMI-Lab/SlotMIM))

* **[2025-05]** InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning. ([Paper](https://arxiv.org/abs/2505.13888), [Website](https://koorye.github.io/proj/Inspire))

* **[2025-05]** ChatVLA-2: Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge. ([Paper](https://arxiv.org/abs/2505.21906), [Website](https://chatvla-2.github.io))

* **[2025-05]** Unveiling the Potential of Vision-Language-Action Models with Open-Ended Multimodal Instructions. ([Paper](https://arxiv.org/abs/2505.11214))

* **[2025-06]** CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding. ([Paper](https://arxiv.org/abs/2506.13725), [Website](https://irpn-eai.github.io/CEED-VLA), [Code](https://github.com/OpenHelix-Team/CEED-VLA))

* **[2025-08]** GeoVLA: Empowering 3D Representations in Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2508.09071), [Website](https://linsun449.github.io/GeoVLA))

### Enhanced Representation for Manipulation

* **[2024-02]** Vision-Language Models Provide Promptable Representations for Reinforcement Learning. ([Paper](https://arxiv.org/abs/2402.02651), [Website](https://pr2l.github.io), [Code](https://github.com/pr2l/pr2l.github.io/blob/master/static/notebooks/PR2LExample.ipynb))

* **[2024-03]** Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics. ([Paper](https://arxiv.org/abs/2403.19578))

* **[2024-05]** Pre-trained Text-to-Image Diffusion Models Are Versatile Representation Learners for Control. ([Paper](https://arxiv.org/abs/2405.05852))

* **[2024-12]** TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies. ([Paper](https://arxiv.org/abs/2412.10345), [Website](https://tracevla.github.io), [Code](https://github.com/umd-huang-lab/tracevla))

* **[2025-01]** SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model. ([Paper](https://arxiv.org/abs/2501.15830), [Website](https://spatialvla.github.io), [Code](https://github.com/SpatialVLA/SpatialVLA))

* **[2025-02]** BFA: Best-Feature-Aware Fusion for Multi-View Fine-grained Manipulation. ([Paper](https://arxiv.org/abs/2502.11161))

* **[2025-02]** VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation. ([Paper](https://arxiv.org/abs/2502.02175), [Website](https://vla-cache.github.io), [Code](https://github.com/siyuhsu/vla-cache))

* **[2025-02]** VLAS: Vision-Language-Action Model With Speech Instructions For Customized Robot Manipulation. ([Paper](https://arxiv.org/abs/2502.13508))

* **[2025-02]** ObjectVLA: End-to-End Open-World Object Manipulation Without Demonstration. ([Paper](https://arxiv.org/abs/2502.19250), [Website](https://objectvla.github.io))

* **[2025-02]** DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping. ([Paper](https://arxiv.org/abs/2502.20900), [Website](https://dexgraspvla.github.io), [Code](https://github.com/Psi-Robot/DexGraspVLA))

* **[2025-03]** OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction. ([Paper](https://arxiv.org/abs/2503.03734), [Website](https://ottervla.github.io), [Code](https://github.com/Max-Fu/otter))

* **[2025-03]** RoboFlamingo-Plus: Fusion of Depth and RGB Perception with Vision-Language Models for Enhanced Robotic Manipulation. ([Paper](https://arxiv.org/abs/2503.19510))

* **[2025-05]** VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation. ([Paper](https://arxiv.org/abs/2505.09577), [Website](https://sites.google.com/view/vtla))

* **[2025-05]** 3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks. ([Paper](https://arxiv.org/abs/2505.05800), [Website](https://3d-cavla.github.io), [Code](https://github.com/vineet2104/3dcavla))

* **[2025-05]** ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation. ([Paper](https://arxiv.org/abs/2505.22159), [Website](https://sites.google.com/view/forcevla2025))

* **[2025-06]** BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models. ([Paper](https://www.arxiv.org/abs/2506.07961), [Website](https://bridgevla.github.io), [Code](https://github.com/BridgeVLA/BridgeVLA))

* **[2025-06]** CronusVLA: Transferring Latent Motion Across Time for Multi-Frame Prediction in Manipulation. ([Paper](https://arxiv.org/abs/2506.19816), [Website](https://lihaohn.github.io/CronusVLA.github.io), [Code](https://github.com/InternRobotics/CronusVLA))

* **[2025-06]** OG-VLA: 3D-Aware Vision Language Action Model via Orthographic Image Generation. ([Paper](https://arxiv.org/abs/2506.01196), [Website](https://og-vla.github.io))

* **[2025-07]** Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding. ([Paper](https://arxiv.org/abs/2507.00416))

* **[2025-07]** VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback. ([Paper](https://arxiv.org/abs/2507.17294), [Website](https://jxbi1010.github.io/vla-touch-gh-pages), [Code](https://github.com/jxbi1010/VLA-Touch))

* **[2025-08]** GeoVLA: Empowering 3D Representations in Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2508.09071), [Website](https://linsun449.github.io/GeoVLA))

* **[2025-08]** MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2508.19236), [Website](https://shihao1895.github.io/MemoryVLA), [Code](https://github.com/shihao1895/MemoryVLA))

* **[2025-08]** RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2508.02062), [Website](https://ricl-vla.github.io), [Code](https://github.com/ricl-vla/ricl_openpi))

* **[2025-08]** Spatial Traces: Enhancing VLA Models with Spatial-Temporal Understanding. ([Paper](https://arxiv.org/abs/2508.09032), [Website](https://ampiromax.github.io/ST-VLA))

* **[2025-08]** OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing. ([Paper](https://arxiv.org/abs/2508.08706), [Website](https://readerek.github.io/Objtac.github.io))

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

* **[2024-12]** Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression. ([Paper](https://arxiv.org/abs/2412.03293), [Website](https://diffusion-vla.github.io))

* **[2025-01]** FAST: Efficient Action Tokenization for Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2501.09747), [Website](https://www.physicalintelligence.company/research/fast))

* **[2025-01]** Universal Actions for Enhanced Embodied Foundation Models. ([Paper](https://arxiv.org/pdf/2501.10105), [Website](https://2toinf.github.io/UniAct/?utm_source=tldrai), [Code](https://github.com/2toinf/UniAct))

* **[2025-02]** Fine-tuning vision-language-action models: Optimizing speed and success. ([Paper](https://arxiv.org/abs/2502.19645), [Website](https://openvla-oft.github.io), [Code](https://github.com/moojink/openvla-oft))

* **[2025-03]** Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding. ([Paper](https://arxiv.org/abs/2503.02310))

* **[2025-03]** Refined Policy Distillation: From VLA Generalists to RL Experts. ([Paper](https://arxiv.org/abs/2503.05833))

* **[2025-03]** HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model. ([Paper](https://arxiv.org/abs/2503.10631), [Website](https://hybrid-vla.github.io), [Code](https://github.com/PKU-HMI-Lab/Hybrid-VLA))

* **[2025-03]** Efficient Continual Adaptation of Pretrained Robotic Policy with Online Meta-Learned Adapters. ([Paper](https://arxiv.org/pdf/2503.18684))

* **[2025-03]** Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy. ([Paper](https://arxiv.org/abs/2503.19757), [Website](https://robodita.github.io), [Code](https://github.com/RoboDita/Dita))

* **[2025-07]** VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting. ([Paper](https://arxiv.org/abs/2507.05116), [Code](https://github.com/LukeLIN-web/VOTE))

* **[2025-07]** VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers. ([Paper](https://arxiv.org/abs/2507.01016), [Website](https://xiaoxiao0406.github.io/vqvla.github.io), [Code](https://github.com/xiaoxiao0406/VQ-VLA))

* **[2025-08]** Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies. ([Paper](https://arxiv.org/abs/2508.20072))

---

## :memo: Deepening Task Comprehension

This section covers methods that enable agents to better understand and generalize across tasks. Key areas include **human–robot interaction**, where agents learn to interpret and respond to human inputs effectively, and **hierarchical task manipulation**, which enables multi-step reasoning and planning by decomposing complex tasks into structured subtasks.

### Human–Robot-Interaction

* **[2023-10]** What Matters to You? Towards Visual Representation Alignment for Robot Learning. ([Paper](https://arxiv.org/abs/2310.07932))

* **[2024-05]** Hummer: Towards Limited Competitive Preference Dataset. ([Paper](https://arxiv.org/abs/2405.11647))

* **[2024-05]** A Self-Correcting Vision-Language-Action Model for Fast and Slow System Manipulation. ([Paper](https://arxiv.org/abs/2405.17418))

* **[2024-12]** Maximizing Alignment with Minimal Feedback: Efficiently Learning Rewards for Visuomotor Robot Policy Alignment. ([Paper](https://arxiv.org/abs/2412.04835))

* **[2025-03]** Adversarial Data Collection: Human-Collaborative Perturbations for Efficient and Robust Robotic Imitation Learning. ([Paper](https://arxiv.org/abs/2503.11646))

* **[2025-03]** VLA Model-Expert Collaboration for Bi-directional Manipulation Learning. ([Paper](https://arxiv.org/abs/2503.04163), [Website](https://aoqunjin.github.io/Expert-VLA))

* **[2025-03]** RoboCopilot: Human-in-the-loop Interactive Imitation Learning for Robot Manipulation. ([Paper](https://arxiv.org/abs/2503.07771))

* **[2025-04]** Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction. ([Paper](https://arxiv.org/abs/2504.14588), [Website](https://xwinks.github.io/motion_instruction_for_correction), [Code](https://github.com/GeWu-Lab/Motion-based-Self-Reflection-Framework))

* **[2025-06]** Robotic Policy Learning via Human-assisted Action Preference Optimization. ([Paper](https://arxiv.org/abs/2506.07127), [Website](https://gewu-lab.github.io/hapo_human_assisted_preference_optimization), [Code](https://github.com/bytedance/human_assisted_preference_optimization))

### Hierarchical Task Manipulation

* **[2023-11]** Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning. ([Paper](https://arxiv.org/abs/2311.17842), [Website](https://robot-vila.github.io), [Code]())

* **[2024-07]** Diffusion Augmented Agents: A Framework for Efficient Exploration and Transfer Learning. ([Paper](https://arxiv.org/abs/2407.20798))

* **[2024-07]** Robotic Control via Embodied Chain-of-Thought Reasoning. ([Paper](https://arxiv.org/abs/2407.08693), [Website](https://embodied-cot.github.io), [Code](https://github.com/MichalZawalski/embodied-CoT))

* **[2024-08]** Policy Adaptation via Language Optimization: Decomposing Tasks for Few-Shot Imitation. ([Paper](https://arxiv.org/abs/2408.16228), [Website](https://palo-website.github.io), [Code](https://github.com/vivekmyers/palo))

* **[2024-10]** HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers. ([Paper](https://arxiv.org/abs/2410.05273))

* **[2024-11]** STEER: Flexible Robotic Manipulation via Dense Language Grounding. ([Paper](https://arxiv.org/abs/2411.03409), [Website](https://lauramsmith.github.io/steer))

* **[2024-11]** GRAPE: Generalizing Robot Policy via Preference Alignment. ([Paper](https://arxiv.org/abs/2411.19309), [Website](https://grape-vla.github.io), [Code](https://github.com/aiming-lab/GRAPE))

* **[2024-11]** CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision. ([Paper](https://arxiv.org/abs/2411.00508), [Website](https://clip-rt.github.io), [Code](https://github.com/gicheonkang/clip-rt))

* **[2024-12]** Emma-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning. ([Paper](https://arxiv.org/abs/2412.11974), [Website](https://declare-lab.github.io/Emma-X), [Code](https://github.com/declare-lab/Emma-X))

* **[2024-12]** RoboMatrix: A Skill-centric Hierarchical Framework for Scalable Robot Task Planning and Execution in Open-World. ([Paper](https://arxiv.org/abs/2412.00171), [Website](https://robo-matrix.github.io), [Code](https://github.com/WayneMao/RoboMatrix))

* **[2025-02]** RoboBrain: A Unified Brain Model for Robotic Manipulation from Abstract to Concrete. ([Paper](https://arxiv.org/abs/2502.21257), [Website](https://superrobobrain.github.io), [Code](https://github.com/FlagOpen/RoboBrain))

* **[2025-02]** Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2502.19417), [Website](https://www.pi.website/research/hirobot))

* **[2025-03]** RoboDexVLM: Visual Language Model-Enabled Task Planning and Motion Control for Dexterous Robot Manipulation. ([Paper](https://arxiv.org/abs/2503.01616), [Website](https://henryhcliu.github.io/robodexvlm), [Code]())

* **[2025-03]** DataPlatter: Boosting Robotic Manipulation Generalization with Minimal Costly Data. ([Paper](https://arxiv.org/abs/2503.19516))

* **[2025-05]** LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning. ([Paper](https://arxiv.org/abs/2406.11815), [Website](https://llarva24.github.io), [Code](https://github.com/Dantong88/LLARVA))

* **[2025-05]** OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning. ([Paper](https://arxiv.org/abs/2505.11917), [Website](https://one-two-vla.github.io), [Code](https://github.com/Fanqi-Lin/OneTwoVLA))

* **[2025-05]** Pre-Trained Multi-Goal Transformers with Prompt Optimization for Efficient Online Adaptation. ([Paper](https://openreview.net/forum?id=DHucngOEe3&noteId=w5HGYk50VI))

* **[2025-05]** Training Strategies for Efficient Embodied Reasoning. ([Paper](https://arxiv.org/abs/2505.08243), [Website](https://ecot-lite.github.io))

* **[2025-06]** Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse. ([Paper](https://arxiv.org/abs/2506.07639))

* **[2025-07]** ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning. ([Paper](https://arxiv.org/abs/2507.16815), [Website](https://jasper0314-huang.github.io/thinkact-vla))

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

* **[2024-09]** Lifelong Autonomous Improvement of Navigation Foundation Models in the Wild. ([Paper](https://openreview.net/forum?id=vBj5oC60Lk), [Website](https://kylestach.github.io/lifelong-nav-rl), [Code](https://github.com/kylestach/lifelong-nav-rl))

* **[2024-10]** Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance. ([Paper](https://arxiv.org/abs/2410.13816), [Website](https://nakamotoo.github.io/V-GPS), [Code](https://github.com/nakamotoo/V-GPS))

* **[2024-10]** GRAPPA: Generalizing and Adapting Robot Policies via Online Agentic Guidance. ([Paper](https://arxiv.org/abs/2410.06473))

* **[2024-12]** Policy Agnostic RL: Offline RL and Online RL Fine-Tuning of Any Class and Backbone. ([Paper](https://arxiv.org/abs/2412.06685), [Website](https://policyagnosticrl.github.io), [Code](https://github.com/MaxSobolMark/PolicyAgnosticRL))

* **[2024-12]** Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model
. ([Paper](https://arxiv.org/abs/2412.13630), [Website](https://policydecorator.github.io), [Code](https://github.com/tongzhoumu/policy_decorator))

* **[2024-12]** RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning. ([Paper](https://arxiv.org/abs/2412.09858), [Website](https://generalist-distillation.github.io), [Code](https://generalist-distillation.github.io/RLDG))

* **[2025-01]** FDPP: Fine-tune Diffusion Policy with Human Preference. ([Paper](https://arxiv.org/abs/2501.08259))

* **[2025-03]** SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning. ([Paper](https://arxiv.org/abs/2503.03480), [Website](https://safevla.github.io), [Code](https://github.com/safevla/safevla))

* **[2025-05]** Lifelong Autonomous Improvement of Navigation Foundation Models in the Wild. ([Paper](https://openreview.net/forum?id=vBj5oC60Lk), [Code](https://github.com/kylestach/lifelong-nav-rl))

* **[2025-05]** What Can RL Bring to VLA Generalization? An Empirical Study. ([Paper](https://arxiv.org/abs/2505.19789), [Website](https://rlvla.github.io), [Code](https://github.com/gen-robot/RL4VLA))

* **[2025-05]** VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning. ([Paper](https://arxiv.org/abs/2505.18719), [Code](https://github.com/GuanxingLu/vlarl))

* **[2025-05]** Interactive Post-Training for Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2505.17016), [Website](https://ariostgx.github.io/ript_vla), [Code](https://github.com/Ariostgx/ript-vla))

* **[2025-05]** ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning. ([Paper](https://arxiv.org/abs/2505.07395), [Code](https://github.com/COST-97/reinboT))

* **[2025-05]** RFTF: Reinforcement Fine-tuning for Embodied Agents with Temporal Feedback. ([Paper](https://arxiv.org/abs/2505.19767))

* **[2025-06]** Inference-Time Alignment via Hypothesis Reweighting. ([Paper](https://openreview.net/forum?id=tl2nXqQSQJ))

* **[2025-06]** Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics. ([Paper](https://arxiv.org/abs/2506.00070))

* **[2025-06]** TGRPO: Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization. ([Paper](https://arxiv.org/abs/2506.08440))

* **[2025-07]** Behavioral Exploration: Learning to Explore via In-Context Adaptation. ([Paper](https://arxiv.org/abs/2507.09041))

* **[2025-07]** Reinforcement Learning with Action Chunking. ([Paper](https://arxiv.org/abs/2507.07969))

* **[2025-08]** CO-RFT: Efficient Fine-Tuning of Vision-Language-Action Models through Chunked Offline Reinforcement Learning. ([Paper](https://arxiv.org/abs/2508.02219))

* **[2025-09]** VLA Model Post-Training via Action-Chunked PPO and Self Behavior Cloning. ([Paper](https://arxiv.org/abs/2509.25718))

### Visual Interaction Prediction

* **[2023-12]** Unleashing large-scale video generative pre-training for visual robot manipulation. ([Paper](https://arxiv.org/abs/2312.13139), [Website](https://gr1-manipulation.github.io), [Code](https://github.com/bytedance/GR-1))

* **[2024-03]** MineDreamer: Learning to Follow Instructions via Chain-of-Imagination for Simulated-World Control. ([Paper](https://arxiv.org/abs/2403.12037), [Website](https://sites.google.com/view/minedreamer/main), [Code](https://github.com/Zhoues/MineDreamer))

* **[2024-06]** Learning Manipulation by Predicting Interaction. ([Paper](https://arxiv.org/abs/2406.00439), [Website](https://opendrivelab.com/MPI), [Code](https://github.com/OpenDriveLab/MPI))

* **[2024-06]** DISCO: Language-Guided Manipulation with Diffusion Policies and Constrained Inpainting. ([Paper](https://arxiv.org/abs/2406.09767), [Website](https://disco2025.github.io))

* **[2024-06]** Generate Subgoal Images before Act: Unlocking the Chain-of-Thought Reasoning in Diffusion Model for Robot Manipulation with Multimodal Prompts. ([Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ni_Generate_Subgoal_Images_before_Act_Unlocking_the_Chain-of-Thought_Reasoning_in_CVPR_2024_paper.html), [Website](https://cotdiffusion.github.io))

* **[2024-07]** VLMPC: Vision-Language Model Predictive Control for Robotic Manipulation. ([Paper](https://arxiv.org/abs/2407.09829), [Code](https://github.com/ppjmchen/vlmpc))

* **[2024-07]** Multimodal Diffusion Transformer: Learning Versatile Behavior from Multimodal Goals. ([Paper](https://arxiv.org/abs/2407.05996), [Website](https://intuitive-robots.github.io/mdt_policy), [Code](https://github.com/intuitive-robots/mdt_policy))

* **[2024-07]** Generative Image as Action Models. ([Paper](https://arxiv.org/abs/2407.07875), [Website](https://genima-robot.github.io), [Code](https://github.com/MohitShridhar/genima))

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

* **[2025-03]** CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models. ([Paper](https://arxiv.org/abs/2503.22020), [Website](https://cot-vla.github.io))

* **[2025-03]** DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation. ([Paper](https://arxiv.org/abs/2503.16806), [Website](https://pku-epic.github.io/DyWA), [Code](https://github.com/jiangranlv/DyWA))

* **[2025-04]** Vision-Language Model Predictive Control for Manipulation Planning and Trajectory Generation. ([Paper](https://arxiv.org/abs/2504.05225), [Code](https://github.com/ppjmchen/vlmpc))

* **[2025-05]** FLARE: Robot Learning with Implicit World Modeling. ([Paper](https://arxiv.org/abs/2505.15659), [Website](https://research.nvidia.com/labs/gear/flare), [Code](http://github.com/nvidia/flare))

* **[2025-06]** WorldVLA: Towards Autoregressive Action World Model. ([Paper](https://arxiv.org/abs/2506.21539), [Code](https://github.com/alibaba-damo-academy/WorldVLA))

* **[2025-07]** DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge. ([Paper](https://arxiv.org/abs/2507.04447), [Website](https://zhangwenyao1.github.io/DreamVLA), [Code](https://github.com/Zhangwenyao1/DreamVLA))

* **[2025-07]** EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos. ([Paper](https://arxiv.org/abs/2507.12440), [Website](https://rchalyang.github.io/EgoVLA))

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

## :clipboard: Survey

* **[2023-12]** Foundation Models in Robotics: Applications, Challenges, and the Future. ([Paper](https://arxiv.org/abs/2312.07843), [Code](https://github.com/robotics-survey/Awesome-Robotics-Foundation-Models))

* **[2023-12]** Toward General-Purpose Robots via Foundation Models: A Survey and Meta-Analysis. ([Paper](https://arxiv.org/abs/2312.08782))

* **[2024-02]** Real-World Robot Applications of Foundation Models: A Review. ([Paper](https://arxiv.org/abs/2402.05741))

* **[2024-05]** A Survey on Vision-Language-Action Models for Embodied AI. ([Paper](https://arxiv.org/abs/2405.14093), [Code](https://github.com/yueen-ma/Awesome-VLA))

* **[2025-03]** A Taxonomy for Evaluating Generalist Robot Policies. ([Paper](https://arxiv.org/abs/2503.01238))

* **[2025-05]** Vision-Language-Action Models: Concepts, Progress, Applications and Challenges. ([Paper](https://arxiv.org/abs/2505.04769), [Website](https://vla-survey.github.io))

* **[2025-07]** A Survey on Vision-Language-Action Models: An Action Tokenization Perspective. ([Paper](https://arxiv.org/abs/2507.01925), [Code](https://github.com/Psi-Robot/Awesome-VLA-Papers))

---

## :black_nib: Contributing

We welcome contributions from the community! Whether it's adding new papers, sharing code, or improving documentation, your input helps make this a valuable resource for everyone!

---

## :pushpin: BibTeX

To cite this repository in your research, please use the following BibTeX entry:

```bibtex
@article{xiang2025parallels,
  title={Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends},
  author={Xiang, Tian-Yu and Jin, Ao-Qun and Zhou, Xiao-Hu and Gui, Mei-Jiang and Xie, Xiao-Liang and Liu, Shi-Qi and Wang, Shuang-Yi and Duan, Sheng-Bin and Xie, Fu-Chao and Wang, Wen-Kai and others},
  journal={arXiv preprint arXiv:2506.20966},
  year={2025}
}
```
