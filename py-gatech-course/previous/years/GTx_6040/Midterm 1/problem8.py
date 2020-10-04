
# coding: utf-8

# # Problem 8: Balancing Act
# 
# "Balancing a seesaw with Python data structures."

# **Background.** Suppose there is a plank of wood pivoted to a point so that it acts as a seesaw. There are weights attached to it at different locations. This module deals with predicting the seesaw behaviour given the distributed weights and pivot point. Let's illustrate it with an example.

# Suppose there are 6 friends sitting on the see-saw. Let's "number" these friends starting at 1, 2, ..., up to 6. Let the weight of each of the friends be 56, 52, 48, 64, 52, 70 (from left to right). The position of the pivot on which the seesaw rests is denoted by the position at which 0 occurs. Such a system is illustrated by following list:

# In[1]:


w = [56, 52, 48, 0, 64, 52, 70]


# One can use rotational concepts to calculate the moment generated by each friend. The equation for the moment of friend $i$ is given by $M_i$, where
# 
# $M_i = F_i * r_i.$
# 
# Here, $F_i$ is the force exerted by friend $i$ and $r_i$ is the distance of weight $i$ from the pivot.
# 
# For the forces, assume they are the same as the weight. Thus, the force of weight $i=1$ is 56, of weight $i=2$ is 52, and weight $i=6$ is 70.
# 
# For the distances, let $r_i$ measure the absolute difference between the index of weight $i$ in the list and the index of the pivot. Thus, in above example, the amount of rotational "force" generated by each of the friends would be calculated as follows:

# In[2]:


Friend_1 = 56*3
Friend_2 = 52*2
Friend_3 = 48*1

Friend_4 = 64*1
Friend_5 = 52*2
Friend_6 = 70*3


# The seesaw will be **balanced** if the amount of rotational forces on each side of the pivot are equal. Therefore, moments generated by individual weights on each side needs to be added and compared to check if seesaw will be balanced or tilted. Please run the cell below to check the behaviour for the seesaw whose weights are given by `w` above.

# In[4]:


#Run this cell

left_moment = Friend_1 + Friend_2 + Friend_3
right_moment = Friend_4 + Friend_5 + Friend_6

print('<---#### Rotation Force measures ####--->')
print('Left_moment: ', left_moment, 'Right_moment: ', right_moment)

if left_moment > right_moment:
    print("The seesaw tilts to the left")
elif right_moment > left_moment:
    print("The seesaw tilts to the right")
else:
    print("The seesaw stays balanced")


# Since the sum of rotational forces on the right side of the pivot is greater than that on the left side, the seesaw tilts towards the right.

# Here is one more example to make the calculation clear:

# In[5]:


W = [53, 76, 87, 54, 0, 76, 52, 67]


# In[6]:


#rotational force of all elements will be :
F_1 = 53 * 4
F_2 = 76 * 3
F_3 = 87 * 2
F_4 = 54 * 1
F_6 = 76 * 1
F_7 = 52 * 2
F_8 = 67 * 3


# In[7]:


left_moment = F_1 + F_2 + F_3 + F_4
right_moment = F_6 + F_7 + F_8

print('<---#### Rotation Force measures ####--->')
print('Left_moment: ', left_moment, 'Right_moment: ', right_moment)

if left_moment > right_moment:
    print("The seesaw tilts to the left")
elif right_moment > left_moment:
    print("The seesaw tilts to the right")
else:
    print("The seesaw stays balanced")


# **Exercise 0** (2 points)
# 
# For a given Python list that represents weights attached to a plank at the indexed positions and the pivot position (denoted by zero), write a function - `get_moment(weights, ordered_pos)` which returns the rotational force of an element with respect to the pivot. Your function should take as input two elements, the list of weights (`weights`) and the position of a target element in `weights` (i.e., a list index, `ordered_pos`). It should return the individual contribution of the weight `weights[ordered_pos]` to the rotational force.
# 
# You can assume that the list of weights will always have exactly one pivot, i.e., one and only one element as 0, and all other elements will be greater than zero. If the pivot position is passed as an argument, its associated force will be 0 (Recall $M_i = F_i * r_i$ and $r_i$ will be 0).

# In[8]:


def get_moment(weights, ordered_pos):
    assert ordered_pos < len(weights)
    ### BEGIN SOLUTION
    pivot_pos = weights.index(0)
    arm = abs(pivot_pos - ordered_pos)
    w = weights[ordered_pos]
    return w*arm


# In[9]:


## Test cell: `single_moment`

def check_moment(w, p, v):
    msg = "<-----Calculating moment for index {} in weights----->".format(p, w)
    print(msg)
    assert get_moment(w, p)  == v 
    print("Passed: Correct moment calculated")
    
    
check_moment([4, 3, 5, 10, 22, 0, 8, 12, 32], 1, 12)
check_moment([0, 3, 5, 10, 22, 13, 8, 12, 32], 4, 88)
check_moment([11, 3, 5, 10, 22, 13, 8, 12, 0],4, 88 )

print("\n(Passed!)")


# **Exercise 1** (3 points)
# 
# Write a function `sum_moment(weights, side)` that returns the total rotational force around the pivot on a given side. Your function should accept two parameters, `weight` and `side`. The parameter `weight` represents the list of ordered weights and the parameter `side` is one of two values, either the string `'left'` or the string `'right'`, which represents whether we want the total sum of rotational forces to the left of the pivot or the right, respectively. Refer to the initial discussion on how to calculate the value.

# In[10]:


def sum_moment(weights, side):
    ### BEGIN SOLUTION
    if side == 'left':
        w = weights
    else:
        w = weights[::-1]
    pivot_position = w.index(0)
    total = 0
    for i in range(pivot_position):
        total += get_moment(w, i)
    return total


# In[11]:


## Test cell: `net_moment`

def check_total_moment(w, s, v):
    msg = "<-----Calculating total moment on {} side in weights----->".format(s, w)
    print(msg)
    assert sum_moment(w, s)  == v 
    print("Passed: Correct moment calculated")
    
check_total_moment([4, 3, 5, 10, 22, 0, 8, 12, 32], 'left', 89)
check_total_moment([4, 3, 5, 10, 22, 0, 8, 12, 32], 'right', 128)
check_total_moment([0, 3, 5, 10, 22, 13, 8, 12, 32], "right", 584)
check_total_moment([0, 3, 5, 10, 22, 13, 8, 12, 32], "left", 0)
check_total_moment([11, 3, 5, 10, 22, 13, 8, 12, 0],'left', 344)
check_total_moment([11, 3, 5, 10, 22, 13, 8, 12, 0],'right', 0)

print("\n(Passed!)")


# **Exercise 2** (2 point)
# 
# Write a function `get_tilt(weights)`, which determines whether the seesaw tilts to the left, tilts to the right, or stays balanced. The function must only take the `weights` list as the input parameter. It should return one of three strings, `"left"`, `"right"`, or `"balanced"`, depending on the tilt of the seesaw.

# In[12]:


def get_tilt(weights):
    ### BEGIN SOLUTION
    moment_left = sum_moment(weights, "left")
    moment_right = sum_moment(weights, "right")
    if moment_left > moment_right:
        tilt = "left"
    elif moment_left < moment_right:
        tilt = "right"
    else:
        tilt = "balanced"
    return tilt


# In[13]:


## Test cell: `tilt_direction`

def check_tilt(w, v):
    msg = "<-----Finding tilt of the weights {}----->".format(w)
    print(msg)
    assert get_tilt(w)  == v 
    print("Passed: Correct Tilt direction")

check_tilt([4, 3, 5, 10, 22, 0, 8, 12, 32], 'right')
check_tilt([4, 13, 5, 10, 22, 0, 8, 12, 32], 'left') 
check_tilt([0, 13, 5, 10, 22, 11, 8, 12, 32], 'right')
check_tilt([4, 13, 5, 10, 22, 0, 15, 12, 30], 'balanced')

print("\n(Passed!)")


# **Exercise 3** (3 points)
# 
# Knowing that a given list of weights is tilted towards one side, suppose we want to know the minimum weight that must be added to the opposite side to balance the seesaw.
# 
# You can think of the problem in the following way. To balance the seesaw, you can add weights to different positions on the opposite end. If you were to add only a single weight at any position on the opposite side, what is the minimum it should be?
# 
# If the seesaw is already balanced, your code must retun 0 because no more weight needs to be added.
# 
# > Hint: Think of how to maximize the moment for a given value of force.

# In[14]:


def add_minimum_weight(weights):
    ### BEGIN SOLUTION
    tilt = get_tilt(weights)
    moment_left = sum_moment(weights, "left")
    moment_right = sum_moment(weights, "right")
    pivot_index = weights.index(0)
    if tilt == "left":
        arm = len(weights) - pivot_index -1
    else:
        arm = pivot_index
    moment_diff = abs(moment_right - moment_left)
    return moment_diff/arm


# In[15]:


# Test cell: `minimum_weight`
import random
import numpy as np

def build_plank():
    length = random.randint(3, 20)
    pivot = random.randint(1, length-2)
    plank = []
    for i in range(length):
        plank.append(np.round(random.random()*100 + 1, 2))
    plank[pivot] = 0
    return plank
        

def test_minimum_weight():
    w = build_plank()
    msg = "<-----Finding minimum weight that can be added on {} to balance it----->".format(w)
    print(msg)
    w_min = add_minimum_weight(w)
    moment_left = sum_moment(w, "left")
    moment_right = sum_moment(w, "right")
    tilt = get_tilt(w)
    pivot = w.index(0)
    l = len(w)
    w_c = w.copy()
    if tilt == "left":
        w_c[l-1] += w_min
    elif tilt == "right":
        w_c[0] += w_min
    new_tilt = get_tilt(w_c)
#     print(w_min)
    assert abs(sum_moment(w_c, "left") - sum_moment(w_c, "right")) <= 10e-10
    print("floating point error is {}".format(abs(sum_moment(w_c, "left") - sum_moment(w_c, "right"))))
    print("Passed: Correct weight identified")
    
n_tests = 10
for t in range(n_tests):
    test_minimum_weight()


# **Fin!** This cell marks the end of this part. Don't forget to save, restart and rerun all cells, and submit it. When you are done, proceed to other parts.