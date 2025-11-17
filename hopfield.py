import numpy as np

class Hopfield:

    def train(self, patterns, height, width):
        patterns = np.array(patterns)
        nums_neurons = height * width
        W = np.zeros((nums_neurons, nums_neurons))

        for p in patterns:
            W += np.outer(p, p)

        np.fill_diagonal(W, 0)
        W /= len(patterns)

        self.W = W

    @staticmethod
    def sign(x):
        return np.where(x >= 0, 1, -1)

    def update_sync(self, state):
        return Hopfield.sign(self.W @ state)

    def update_async(self, state, iters=10):
        state = state.copy()
        for _ in range(iters):
            for i in range(self.n):
                s = np.dot(self.W[i], state)
                # nie wiem czy to jest ok to wyglada podejrzanie tez slabo otwarza
                state[i] = 1 if s >= 0 else -1
        return state

    def energy(self, state):
        return -0.5 * state.T @ self.W @ state

    # nie dziala idk, jakos obraz najlepeij dziala po jednej iteracji
    def run(self, state, max_iters=20):
        state = state.copy()
        prev_E = np.inf

        for _ in range(max_iters):
            state = self.update_sync(state)
            E = self.energy(state)

            if abs(prev_E - E) < 1e-6:
                break
            prev_E = E

        return state

    def run_once(self, state):
        return self.hopfield_sign(self.W @ state) 
        
    def hopfield_sign(self, x): 
        return np.where(x >= 0, 1, -1)

