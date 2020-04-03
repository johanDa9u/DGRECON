import numpy as np
import random


class NN:
    def __init__(self, dimensions=[], sigma=1, weights=None, bias=None):
        """
        Initialise le réseau de neurone
        """
        if weights is not None and bias is not None:
            self.weights = weights
            self.bias = bias
        else:
            self.random_initialization(dimensions, sigma)

    def random_initialization(self, dimensions, sigma):
        """
        Initialise les matrices weights et bias à des valeurs aléaroires
        de distribution gaussienne et d'écart type sigma. 
        - dimensions doit être un vecteur de taille N+1 où N est le nombre de couches
        - sigma>0 est l'écart type de l'initialization aléatoire 
        """
        self.weights = []
        self.bias = []
        for i in range(0, len(dimensions)-1):
            self.weights.append(np.random.normal(
                0, sigma, (dimensions[i+1], dimensions[i])))
            self.bias.append(np.random.normal(0, sigma, (dimensions[i+1], 1)))

    def forward_prediction(self, x):
        """
        Retourne un vecteur a d_n "loss"
        """
        for b, w in zip(self.bias, self.weights):
            x = relu(np.dot(w, x)+b)
        return softmax(x)

    def backprop(self, x, y):

        # Initialisation des gradiants
        gradient_poids = [np.zeros(w.shape) for w in self.weights]
        gradient_biais = [np.zeros(b.shape) for b in self.bias]

        a = x
        a_liste = [x]
        z_liste = []  # cette liste contiendra tous les W*h +b
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, a)+b
            z_liste.append(z)
            a = relu(z)
            a_liste.append(a)
        d = c_a(a_liste[-1], y) * der_relu(z_liste[-1])
        gradient_biais[-1] = d
        gradient_poids[-1] = np.dot(d, a_liste[-2].transpose())
        for l in range(2, len(self.weights)+1):
            z = z_liste[-l]
            sp = der_relu(z)
            d = np.dot(self.weights[-l+1].transpose(), d) * sp
            gradient_biais[-l] = d
            gradient_poids[-l] = np.dot(d, a_liste[-l-1].transpose())

        return(gradient_poids, gradient_biais)

    # Teste la precision sur test_data en renvoyant le nombre de de predictions correctes
    def performance(self, test_data):
        compteur = 0
        for x, y in test_data:
            if(np.argmax(self.forward_prediction(x)) == y):
                compteur += 1
        return compteur

    # Mets a jour les poids et biais du NN en utilisant la retropropagation de gradient
    def mise_a_jour_poids(self, mini_batch, alpha):
        grad_bias = [np.zeros(b.shape) for b in self.bias]
        grad_weights = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_grad_weights, delta_grad_bias = self.backprop(x, y)
            grad_bias = [nb+dnb for nb, dnb in zip(grad_bias, delta_grad_bias)]
            grad_weights = [nw+dnw for nw, dnw in zip(grad_weights, delta_grad_weights)]
        self.weights = [w-(alpha/len(mini_batch))*nw for w, nw in zip(self.weights, grad_weights)]
        self.biases = [b-(alpha/len(mini_batch))*nb for b, nb in zip(self.bias, grad_bias)]

    def train_NN(self, X_train, Y_train, alpha=0.01):
        """X and Y are vectors"""
        epochs = 30  # nombre de fois que l'on va apprendre
        mini_batch_size = 3 # on va separer ici notre jeu d'entrainement en partitions.
        # on separe nos donnés 70% train et 30% test
        lt = data_to_tuple2(X_train, Y_train) # mise en forme de X_train et Y_train, voir data_to_tuple plus en bas
        lt2 = data_to_tuple(X_train, Y_train)
        l1 = lt[:7000]
        l2 = lt2[7000:10000]
        training_data = l1
        test_data = l2
        n_test = len(test_data)
        n_test = len(test_data)
        n = len(training_data)
        print("Demarrage de l'entrainement...")
        for j in range(epochs):
            random.shuffle(training_data)  # On mélange notre train
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                # On met a jour les pods avec la retropropagation pour chaque partition du training data
                self.mise_a_jour_poids(mini_batch, alpha)
            # evaluation des performances sur chaque tour d'epoch
            print("traitement ", j, " sur ", epochs, " : ", self.performance(test_data), " correctes / ", n_test, " au total")

# On utilise softmax sur l'output final
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Derive de la fonction de cout
# On utilise ici l'entropie croisée comme fonction de cout
def c_a(a, y):
    res = []
    val_y = np.argmax(y)
    for i in range(0, len(a)):
        if(i == val_y):
            res.append(-1 + (np.exp(a[i])/np.sum(np.exp(a), axis=0)))
        else:
            res.append(np.exp(a[i])/np.sum(np.exp(a), axis=0))
    return np.asarray(res)

# On utilise ReLu comme fonction d'activation
def relu(x):
    return np.maximum(0, x)

# dérivé de la fonction relu
def der_relu(Z):  

    res = []
    for x in Z:
        res2 = []
        for y in x:
            if y < 0:
                res2.append(0)
            if y > 0:
                res2.append(1)
        res.append(res2)
    return np.asarray(res)


# converti x_train et y_train en une liste de tuple representant les entrées et les sorties attendues
def data_to_tuple(x_train, y_train):
    tuple = []
    for i, j in zip(x_train, y_train):
        tuple.append((np.array(i).reshape(-1, 1), j))
    return tuple

# converti x_train et y_train en une liste de tuple representant les entrées et les sorties attendues (y vectorisés)
#data_to_tuple2  sera donc utilise pour l entrainement alors que data_to_tuple sera utilise pour l evaluation des performances
def data_to_tuple2(x_train, y_train):
    tuple = []
    for i, j in zip(x_train, y_train):
        y = [0] * 10
        y[j] = 1
        tuple.append((np.array(i).reshape(-1, 1), np.array(y).reshape(10, -1)))
    return tuple


file = np.load('MNIST.npz')
X = file['X']
Y = file['Y']

# On créer le NN:
net = NN(dimensions=[784, 512, 256, 120, 80, 40, 20, 10], sigma=0.1)
net.train_NN(X, Y)


#On remarque ici que des le premier tour d'epoch, on a un resultat assez correcte.
#Puis le nombre de predictions correcte augmente au fur et a mesure, prouvant ainsi
#une bonne mise a jours des poids et des biais du NN.

#Note: J ai egalement essaye avec d autres parametres, en changeant par exemple la fonction
#d activation par MSE, en utilisant la fonction sigmoide...
#Mais cette combinaison (ReLu + Entropie croisee + Softmax sur l output) semble 
#donner les meilleurs resultats.

#On peut encore jouer sur la taille des partitions (mini_batch_size dans la fonction train) et
#sur la valeure d'epochs.