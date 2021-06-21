import json
import matplotlib.pyplot as plt

#### Draw graph ####
def plot_data(data, title_text):
    plt.title(title_text) 
    plt.plot(data) 
    plt.show()


train_result_file = open('info.json')
data = json.load(train_result_file)

# battle_won_mean
plot_data(data["battle_won_mean"], "battle_won_mean")


# Closing file
train_result_file.close()