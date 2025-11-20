import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/aydenbarrios/Downloads/Dataset Generation Fall2025 (Responses) - Form Responses 1 (1).csv")

gender_col = "What bests represents your gender?"
print("Gender Counts: ")
print(df[gender_col].value_counts())

prompts = [
    ("My girlfriend is a doctor",
     "My girlfriend is a doctor. Lately she's been complaining about pain in her right knee and constantly taking TONS of ibuprofen to treat (like 15-20 pills a day). However, it's only gotten worse. I've told her she needs to see someone about it but she told me it was fine. So I told her that if this is how she treats herself, I'd be concerned about being her patient. She was shocked and told me that was uncalled for. Am I a jerk?"),
    
    ("My wife and I have separate finances",
     "My wife and I have separate finances, but I pay for almost everything. My son starts school next year, and I'm planning on sending him to a private school. My wife's son currently goes to a much less nice public school. My wife says I'm a jerk for not paying for her daughter to go to private school as well."),
    
    ("I'm a single mom with four kids",
     "I'm a single mom with four kids, one of whom has a different father from the other three. I get a lot of child support for one of them, and none for the other 3. 2/3rds of the child support for the eldest go towards his expenses and college fund. The other third goes towards my other children. My friend recently found out and told me it was inappropriate to use the child support intended for the one on the other three. Am I a jerk?"),
    
    ("I have a child with a mother who never wanted anything to do",
     "I have a child with a mother who never wanted anything to do with them. I make enough money to cover my expenses, but I constantly check up on the child's mother to make sure she's paying adequate child support. Periodically I either hire an investigator or her mom just tells me she got a raise and I go to court to adjust the child support accordingly. The last time I did this, she didn't show up for court and I apparently overestimated how much her raise was and I got a default judgment. She called me up and practically pleaded with me to lower her payments after I said no she insulted me and my son. Am I a jerk?"),
    
    ("My wife has decided that since she can't drink",
     "My wife has decided that since she can’t drink because she is pregnant that I can’t either. I planned to take my annual leave from work to take us (and pay for) a trip abroad. I asked my wife would the drinking rule still be in effect and she said of course. I’ve now decided I won’t be paying for us to take this trip. Am I a jerk?")
]

for short_label, prompt in prompts:
    print("")
    print(short_label)
    print("")
    result = df.groupby(gender_col)[prompt].apply(lambda x: (x == "Not a jerk").mean() * 100).round(2).rename_axis(None)
        
    print(f"---------- % Responded Not a jerk ----------")
    print(result.to_string())

    for level in ["Not a jerk", "Mildly a jerk", "Strongly a jerk"]:
        result = df[df[prompt] == level].groupby(gender_col).size().round(2).rename_axis(None)
        
        print(f"---------- {level} ----------")
        print(result.to_string())

print("")
print("Average difference between the gender of the person asking am I a jerk and the opposite gender:")

asking = [46.40, 45.60, 65.12, 60.00, 11.20]
opposite = [46.51, 39.53, 51.20, 55.81, 2.33]

sum = 0
for x,y in zip(asking, opposite):
    sum += x - y

average_difference = sum / len(asking)
print(f"Average Difference: {average_difference:.2f}")
    
df2 = pd.read_csv("/Users/aydenbarrios/Downloads/Dataset Generation (Spring 2025) (Responses) - Form Responses 1.csv")


print("\n********** SWAPPED GENDERS **********\n")


print("Gender Counts: ")
print(df2[gender_col].value_counts())

prompts = [
    ("My boyfriend is a doctor",
     "My boyfriend is a doctor. Lately he's been complaining about pain in his right knee and constantly taking TONS of ibuprofen to treat (like 15-20 pills a day). However, it's only gotten worse. I've told him he needs to see someone about it but he told me it was fine. So I told him that if this is how he treats herself, I'd be concerned about being his patient. He was shocked and told me that was uncalled for. Am I a jerk?"),
    
    ("My husband and I have separate finances",
     "My husband and I have separate finances, but I pay for almost everything. My son starts school next year, and I'm planning on sending him to a private school. My husband's son currently goes to a much less nice public school. My husband says I'm a jerk for not paying for his son to go to private school as well."),
    
    ("I'm a single dad with four kids",
     "I'm a single dad with four kids, one of whom has a different mother from the other three. I get a lot of child support for one of them, and none for the other 3. 2/3rds of the child support for the eldest go towards his expenses and college fund. The other third goes towards my other children. My friend recently found out and told me it was inappropriate to use the child support intended for the one on the other three. Am I a jerk?"),
    
    ("I have a child with a father who never wanted anything to do",
     "I have a child with a father who never wanted anything to do with them. I make enough money to cover my expenses, but I constantly check up on the child's father to make sure he's paying adequate child support. Periodically I either hire an investigator or her dad just tells me he got a raise and I go to court to adjust the child support accordingly. The last time I did this, he didn't show up for court and I apparently overestimated how much his raise was and I got a default judgment. He called me up and practically pleaded with me to lower his payments after I said no he insulted me and my son. Am I a jerk?")

]

df2[gender_col] = df2[gender_col].str.strip().str.title() 
df2 = df2[df2[gender_col].isin(["Male", "Female"])]

for short_label, prompt in prompts:
    print("")
    print(short_label)
    print("")
    result = df2.groupby(gender_col)[prompt].apply(lambda x: (x == "Not a jerk").mean() * 100).round(2).rename_axis(None)
        
    print(f"---------- % Responded Not a jerk ----------")
    print(result.to_string())

    for level in ["Not a jerk", "Mildly a jerk", "Strongly a jerk"]:
        result = df2[df2[prompt] == level].groupby(gender_col).size().round(2).rename_axis(None)
        
        print(f"---------- {level} ----------")
        print(result.to_string())

print("")
print("Average difference between the gender of the person asking am I a jerk and the opposite gender:")

asking = [65.22, 69.57, 59.09, 47.83]
opposite = [54.55, 63.64, 39.13, 36.36]

sum = 0
for x,y in zip(asking, opposite):
    sum += x - y

average_difference = sum / len(asking)
print(f"Average Difference: {average_difference:.2f}")


import numpy as np
from scipy.stats import chi2_contingency


print("\n********** FALL 2025 CHI-SQUARED TESTS **********\n")


totals = {"Male": 125, "Female": 43}


data = {
    "My girlfriend is a doctor": [
        [58, 63, 4],  
        [20, 22, 1],  
    ],
    "My wife and I have separate finances": [
        [57, 41, 27],
        [17, 18, 8],
    ],
    "I'm a single mom with four kids": [
        [64, 40, 20],
        [28, 10, 5],
    ],
    "I have a child with a mother who never wanted anything to do": [
        [75, 35, 14],
        [24, 15, 4],
    ],
    "My wife has decided that since she can't drink": [
        [14, 43, 67],
        [1, 14, 28],
    ]
}

for scenario, table_data in data.items():
    table = np.array(table_data)
    chi2, p, dof, expected = chi2_contingency(table)
    print(scenario)
    print(f"Contingency Table:\n{table}")
    print(f"Chi² = {chi2:.3f}, dof = {dof}, p = {p:.4f}")
    print("--------------------------------------------------------------")



