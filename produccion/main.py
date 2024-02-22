from utils import Utils
from models import Models


if __name__ == "__main__":
    
    utils = Utils()
    models = Models() #Se ejecuta constructor
    data = utils.load_from_csv('./in/felicidad.csv')
    #print(data)

    x, y = utils.features_target(data, ['score', 'rank', 'country'], ['score'])

    models.grid_training(x, y)



