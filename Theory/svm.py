import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Points:
    def __init__(self, n=50, margin=1.0, random_state=0):
        self.xlim = (-5, 5)
        self.ylim = (-5, 5)
        self.n = n
        self.margin = margin

        np.random.seed(random_state)

class VPoints(Points):
    def __init__(self, n=250, margin=1.0, random_state=0):
        super().__init__(n, margin, random_state)
        self.margin = .5
        self.slope1 = 1.0
        self.slope2 = -1.0
        self._generate_points()

    def _get_distance1(self, x, y):
        return np.abs(x - y) / np.sqrt(2)

    def _get_distance2(self, x, y):
        return np.abs(x + y) / np.sqrt(2)

    def _generate_points(self):
        x, y, = list(), list()

        while len(x) < self.n:
            _x = np.random.uniform(self.xlim[0] * .95, self.xlim[1] * .95)
            _y = np.random.uniform(self.ylim[0] * .95, self.ylim[1] * .95)
            
            should_add = True

            if _x > 0 and _y > 0 and self._get_distance1(_x, _y) < self.margin:
                should_add = False

            if _x < 0 and _y > 0 and self._get_distance2(_x, _y) < self.margin:
                should_add = False

            if should_add:
                x.append(_x)
                y.append(_y)

        c = [self._get_class(_x, _y) for _x, _y in zip(x, y)]
        self.df = pd.DataFrame({'x': x, 'y': y, 'c': c})
        self.df.c = pd.Categorical(self.df.c)

    def _get_class(self, x, y):
        if y <= 0:
            return -1

        if x >= 0 and y >= x:
            return 1

        if x <= 0 and y >= -x:
            return 1

        return -1

    def plot(self):
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(x='x', y='y', hue='c', palette=['r', 'b'], data=self.df, ax=ax, legend=False)            

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        return fig, ax

    def plot3d(self):
        self.df['z'] = np.abs(self.df.x)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        cp = self.df[self.df.c == 1]
        cn = self.df[self.df.c == -1]
        ax.scatter(cp.x, cp.y, cp.z, c='b')
        ax.scatter(cn.x, cn.y, cn.z, c='r')

        return fig, ax

class CirclePoints(Points):
    def __init__(self, n=250, margin=1.0, random_state=0):
        super().__init__(n, margin, random_state)
        self.radius = (self.xlim[1] - self.xlim[0]) / 2.5
        self.margin = .5
        self._generate_points()

    def _generate_points(self):
        x, y, = list(), list()

        while len(x) < self.n:
            _x = np.random.uniform(self.xlim[0] * .95, self.xlim[1] * .95)
            _y = np.random.uniform(self.ylim[0] * .95, self.ylim[1] * .95)
            r = np.sqrt(_x ** 2 + _y ** 2)

            if np.abs(self.radius - r) >= self.margin:
                x.append(_x)
                y.append(_y)

        c = [self._get_class(_x, _y) for _x, _y in zip(x, y)]
        self.df = pd.DataFrame({'x': x, 'y': y, 'c': c})
        self.df.c = pd.Categorical(self.df.c)

    def _get_class(self, x, y):
        return 1 if np.sqrt(x ** 2 + y ** 2) <= self.radius else -1

    def plot(self):
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(x='x', y='y', hue='c', palette=['r', 'b'], data=self.df, ax=ax, legend=False)            

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        return fig, ax

    def plot3d(self):
        self.df['z'] = np.sqrt(self.df.x ** 2 + self.df.y ** 2)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        cp = self.df[self.df.c == 1]
        cn = self.df[self.df.c == -1]
        ax.scatter(cp.x, cp.y, cp.z, c='b')
        ax.scatter(cn.x, cn.y, cn.z, c='r')

        return fig, ax

class LSPoints(Points):
    def __init__(self, n=50, margin=1.0, random_state=0):
        super().__init__(n, margin, random_state)
        self.slope = np.random.uniform(self.ylim[0], self.ylim[1])
        self.slope /= np.random.uniform(self.xlim[0], self.xlim[1])
        self._generate_points()

    def _generate_points(self):
        x, y = list(), list()

        while len(x) < self.n:
            _x = np.random.uniform(self.xlim[0] * .95, self.xlim[1] * .95)
            _y = np.random.uniform(self.ylim[0] * .95, self.ylim[1] * .95)
            d = self._get_distance(_x, _y)

            if d >= self.margin:
                x.append(_x)
                y.append(_y)

        c = [self._get_class(_x, _y) for _x, _y in zip(x, y)]
        self.df = pd.DataFrame({'x': x, 'y': y, 'c': c})
        self.df.c = pd.Categorical(self.df.c)

    def _get_class(self, x, y):
        return int(np.sign(y - self.slope * x))

    def _get_distance(self, x, y):
        return np.abs(self.slope * x - y) / np.sqrt(self.slope ** 2 + 1)

    def plot(self, draw_separator=False, draw_shifted=False, draw_rotated=False):
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(x='x', y='y', hue='c', palette=['r', 'b'], data=self.df, ax=ax, legend=False)

        if draw_separator:
            xs = [self.xlim[0], self.xlim[1]]
            ys = [self.slope * self.xlim[0], self.slope * self.xlim[1]]
            sns.lineplot(xs, ys, color='k', ax=ax, label='Separating hyperplane')
        
        if draw_shifted:
            shift = (self.ylim[1] - self.ylim[0]) / 100.0
            xs = [self.xlim[0], self.xlim[1]]
            ys = [self.slope * self.xlim[0] + shift, self.slope * self.xlim[1] + shift]
            sns.lineplot(xs, ys, color='g', ax=ax, label='Shifted separating hyperplane')

        if draw_rotated:
            slp = self.slope * np.random.uniform(0.85, 0.9)
            xs = [self.xlim[0], self.xlim[1]]
            ys = [slp * self.xlim[0], slp * self.xlim[1]]
            sns.lineplot(xs, ys, color='g', ax=ax, label='Rotated separating hyperplane')
            
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        return fig, ax

class StripePoints(Points):
    def __init__(self, n=250, margin=1.0, random_state=0):
        super().__init__(n, margin, random_state)
        self.slope = 1.0
        self.margin = margin
        self.intercept1 = self.ylim[0] + (self.ylim[1] - self.ylim[0]) / 4.0
        self.intercept2 = self.ylim[0] + (self.ylim[1] - self.ylim[0]) * 3.0 / 4.0
        self.slope = np.random.uniform(self.ylim[0], self.ylim[1])
        self.slope /= np.random.uniform(self.xlim[0], self.xlim[1])
        self._generate_points()

    def _generate_points(self):
        x, y = list(), list()

        while len(x) < self.n / 3:
            _x = np.random.uniform(self.xlim[0] * .95, self.xlim[1] * .95)
            _y = np.random.uniform(self.ylim[0] * .95, self.ylim[1] * .95)
            d1 = self._get_distance(_x, _y, self.intercept1)
            d2 = self._get_distance(_x, _y, self.intercept2)

            if d1 >= self.margin and d2 >= self.margin:
                if (self.slope * _x - _y + self.intercept1 < 0) and (self.slope * _x - _y + self.intercept2 >= 0):
                    x.append(_x)
                    y.append(_y)

        while len(x) < self.n:
            _x = np.random.uniform(self.xlim[0] * .95, self.xlim[1] * .95)
            _y = np.random.uniform(self.ylim[0] * .95, self.ylim[1] * .95)
            d1 = self._get_distance(_x, _y, self.intercept1)
            d2 = self._get_distance(_x, _y, self.intercept2)

            if d1 >= self.margin and d2 >= self.margin:
                if (self.slope * _x - _y + self.intercept1 >= 0) or (self.slope * _x - _y + self.intercept2 < 0):
                    x.append(_x)
                    y.append(_y)

        c = [self._get_class(_x, _y) for _x, _y in zip(x, y)]

        if c.count(-1) > c.count(1):
            c = [-1 if _c == 1 else 1 for _c in c]

        self.df = pd.DataFrame({'x': x, 'y': y, 'c': c})
        self.df.c = pd.Categorical(self.df.c)

    def _get_distance(self, x, y, q):
        return np.abs(self.slope * x - y + q) / np.sqrt(self.slope ** 2 + 1)

    def _get_class(self, x, y):
        if self.slope * x - y + self.intercept1 >= 0:
            return 1
        elif self.slope * x - y + self.intercept2 >= 0:
            return -1
        else:
            return 1

    def plot(self, plot_separators=False):
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(x='x', y='y', hue='c', palette=['r', 'b'], data=self.df, ax=ax, legend=False)            

        if plot_separators:
            xs = [self.xlim[0], self.xlim[1]]
            ys1 = [x * self.slope + self.intercept1 for x in xs]
            ys2 = [x * self.slope + self.intercept2 for x in xs]
            sns.lineplot(xs, ys1, color='k', label='Sep 1')
            sns.lineplot(xs, ys2, color='k', label='Sep 2')

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        return fig, ax