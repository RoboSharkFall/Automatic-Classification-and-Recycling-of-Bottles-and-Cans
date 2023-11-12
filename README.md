# Automatic Classification and Recycling of Bottles and Cans
This package is an implementation of classifying and recycling bottles and cans automatically based on SVM and robotics. To achieve the automatic recycling of bottles and cans, we must address several key issues: identifying where the bottles and cans are located, determining their types, and figuring out how to pick them up and place them properly. Then, we set this task to 3 parts: Image segmentation, Object classification, picking and placing.

### *Experiment* 
<p align = "center">
<img src="GIF/project_order_2.gif" width="360" height="288"> 
</p>

## 1. Build on ROS
In your [files_name]

```
cd ~/files_name
git clone https://github.com/RoboSharkFall/Automatic_Classification_and_Recycling_of_Bottles_and_Cans.git
```

## 2. Run on Ubuntu
```
run main_bottle_picking.py
```

> * 'functions_bottle_recog.py': Functions used for image segmentation
> * 'functions_bottle_classification.py':Functions used for classifying bottles and cans.
> * 'functions_bottle_picking.py': Functions used for calculating objects' pose.


