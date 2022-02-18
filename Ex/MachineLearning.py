# Среднее арифметическое
import statistics # pip install statistics
print("Среднее: ", statistics.mean([-1,0,4,2,1,2]))
# Медиана
print("Медиана: ", statistics.median([-1,0,4,2,1,2]))
# Дисперсия-минимальное отклонение, Отклонение (среднее квдратическое отклонение)
print("Отклонение: ", statistics.variance([1,1,1,1,1,1]))
print("Отклонение: ", statistics.pvariance([2,3,4,5,5,6,8,8,9]))

# Корреляция: Укажите два месяца, которым соответствует наименьшее (по модулю) значение
# (КК) коэффициента корреляции Р=«год рождения студента» и Q=«возраст студента»
import pandas as pd # pip install pandas
data = pd.DataFrame({"birthday" : pd.date_range(start='1/1/1989', end='1/1/2006', periods=100)})
for num, mn in enumerate("January February March April May June July August September October November December".split(), start=1):
    data[mn] = data.birthday.apply(lambda x: int((pd.to_datetime(f'2021-{num}-04 00:00:00') - x).days/365.2 ))
data['year'] = data.birthday.apply(lambda x: x.year)
print("Корреляция наименьшее, Регрессия-наибольший разброс/расхождение: ", data.corr().year.sort_values(ascending=False))


# *** Восстановление пропущенных значений ***
# Евклидово расстояние между ними двумя векторами значений признаков Р1=(0,1,2) Р2=(2,1,0) равно:
import math
point_1 = (0, 1, 2)
point_2 = (2, 1, 0)
distance = math.dist(point_1, point_2)
print("Euclidean distance from P1 to P2: ",distance)
# Вариант решения №2
import numpy as np
point_1 = np.array((0, 1, 2))
point_2 = np.array((2, 1, 0))
square = np.square(point_1-point_2)
sum_square = np.sum(square)
distance = np.sqrt(sum_square)
print("Евклидово расстояние между Р1 и Р2 равно:", distance)
# Вариант решения №3,4
op1=np.sqrt(np.sum(np.square(point_1-point_2)))
op2=np.linalg.norm(point_1-point_2)
print("Еще Евклидово:", op1)
print("Еще Евклидово:", op2)
# Расстояние в метрике Манхэттен между ними равно:
vector1 = np.array([0,1,2])
vector2 = np.array([2,1,0])
op3=np.sum(np.abs(vector1-vector2))
op4=np.linalg.norm(vector1-vector2,ord=1)
print("Расстояние в метрике Манхэттен:", op3)
print("Еще расстояние в метрике Манхэттен:", op4)
# Расстояние между ними в max-метрике (Расстояние Чебышева) равно:
vector1 = np.array([0,1,2])
vector2 = np.array([2,1,0])
op5=np.abs(vector1-vector2).max()
op6=np.linalg.norm(vector1-vector2,ord=np.inf)
print("Расстояние между ними в max-метрике (Расстояние Чебышева):", op5)
print("Еще Расстояние между ними в max-метрике (Расстояние Чебышева):", op6)

# Нормализуем Р=(1,0,5,2,2) вектор по формуле, использующей минимальное и максимальное значение признака Р
print("Нормализуем мин и макс значение признака Р:")
P = [1,0,5,2,2]
for i in P:
    i = (i - min(P)) / (max(P) - min(P))
    print(i, end=', ')
# Нормализовать данные в Python. Чаще всего масштабирование данных изменяется в диапазоне от 0 до 1
from sklearn import preprocessing # pip install preprocessing
import numpy as np
x_array = np.array([1,0,5,2,2])
normalized_arr = preprocessing.normalize([x_array])
print("ИЛИ еще так: ", normalized_arr)

# Требуется оценить, какую оценку поставит Саша фильму «Гарри Поттер». (поставили Вася 3, Петя 4, Маша 5)
# Сделаем это с помощью метрики Манхэттен (для простоты вычислений данные в таблице нормировать не нужно).
# Для этого подсчитаем расстояния от Саши до других людей, используя информацию из первых 3 столбцов.
# Чему равна ожидаемая оценка для «Гарри Поттера» (округлить до одного знака после запятой)?
# Расстояние в метрике Манхэттен между ними равно:
Vasya = np.array([5,5,5])
Petya = np.array([5,3,4])
Masha = np.array([2,5,3])
Sasha = np.array([3,4,4])
sv=np.sum(np.abs(Sasha-Vasya))
sp=np.sum(np.abs(Sasha-Petya))
sm=np.sum(np.abs(Sasha-Masha))
otv = 1 / (1/sv + 1/sp + 1/sm) * (3/sv + 4/sp + 5/sm)
print("Саша и Вася в метрике Манхэттен:", sv)
print("Саша и Петя в метрике Манхэттен:", sp)
print("Саша и Маша в метрике Манхэттен:", sm)
print("Ожидаемая оценка:", otv)

# По строкам отложены наименования товаров, а столбцы — номера заказов
# (в ячейке стоит 1, если товар входит в соответствующий заказ; 0 — в противном случае).
# При построении рекомендательной системы необходимо оценить степень похожести товаров с помощью евклидовой метрики.
# Наиболее похожим (близким) на товар А будет товар С и расстояние между этими товарами равно
import math
A = (1, 0, 1, 0, 1, 0)
C = (1, 1, 0, 1, 1, 0)
distance = math.dist(A, C)
print("Расстояние между этими товарами равно: ",distance)


# *** Поиск выбросов и аномалий ***
# Первая и третья квартиль значений признака Р равны 2, 4 соответственно.
# Какие из следующих значений будут считаться выбросами?
Q1 = 2 # первая квартиль
Q3 = 4 # вторая квартиль
out = (Q1 - 1.5 * (Q3 - Q1))
out2 = (Q3 + 1.5 * (Q3 - Q1))
print("Выбросами будут считаться значения выходящие за пределы от:", out, "до", out2)

# Среднее значение, отклонение и медиана значений признака Р равны 10, 1.1, и 9
# Какие из следующих значений будут выбросами? Не забудьте в процессе решения проверить симметричность выборки.
s = 10
o = 1.1
m = 9
# Определяем интервал по формуле (среднее - 3 * отклонение)  (среднее + 3 * отклонение)
min = s-3*o
max = s+3*o
print("Выбросы за пределами интервала от: ", min, "до", max)
# Если среднее и медиана близки по значениям, то выборка считается симметричной

# В данной задаче выбросы будем искать по следующему правилу:
# «Выбросом будет считаться объект, у которого суммарное расстояние от него до остальных объектов выборки наибольшее».
# Таким образом, в указанной таблице выбросом будет: D
# (при вычислении использовать метрику Манхэттен, нормализацию не проводить),
# сумма расстояний от него до остальных объектов будет равна: 20
A = np.array([1,1,0])
B = np.array([0,2,-1])
C = np.array([2,3,1])
D = np.array([1,0,4])
# посчитать растояние между всеми объектами. Т.е p(АВ), p(АС), p(AD) , p(ВА) и т.д
pАВ = np.sum(np.abs(A-B))
pАС = np.sum(np.abs(A-C))
pAD = np.sum(np.abs(A-D))
pВА = np.sum(np.abs(B-A))
pВС = np.sum(np.abs(B-C))
pВD = np.sum(np.abs(B-D))
pDA = np.sum(np.abs(D-A))
pDB = np.sum(np.abs(D-B))
pDC = np.sum(np.abs(D-C))
print("Расстояние в метрике Манхэттен между ними p(АВ):", pАВ)
print("Расстояние в метрике Манхэттен между ними p(АС):", pАС)
print("Расстояние в метрике Манхэттен между ними p(АD):", pAD)
print("Расстояние в метрике Манхэттен между ними p(ВА):", pВА)
print("Расстояние в метрике Манхэттен между ними p(ВС):", pВС)
print("Расстояние в метрике Манхэттен между ними p(ВD):", pВD)
print("Расстояние в метрике Манхэттен между ними p(DС):", pDA)
print("Расстояние в метрике Манхэттен между ними p(DС):", pDB)
print("Расстояние в метрике Манхэттен между ними p(DС):", pDC)
# Найти сумму расстояний объектов p(AB)+p(AC)+p(AD)=?  и тд
P1 = pАВ+pАС+pAD
P2 = pВА+pВС+pВD
P3 = pDA+pDB+pDC
print("сумма расстояний от него до остальных объектов будет равна: ", P1)
print("сумма расстояний от него до остальных объектов будет равна: ", P2)
print("сумма расстояний от него до остальных объектов будет равна: ", P3)


# *** Кластеризация с помощью графов ***
#Какие объекты попадают в окружность с центром (1,2) и радиусом 2,5   (используется max-метрика)?
#создадим датафрейм с данными
X = np.array([[4, 2], [3, 2], [1, -1], [-1, 1], [0, 4]])
objects_df = pd.DataFrame(data=X, index=['A', 'B', 'C', 'D', 'E'],
                        columns=['P1', 'P2'])
objects_df
#для решения лежит ли точка в пределах окружности, а также в целях тренировки в ООП напишем микро-класс Circle
class Circle():
    def __init__(self, x1_center, x2_center, radius):
        self.x1_center = x1_center
        self.x2_center = x2_center
        self.radius = radius
    def is_in_circle(self, x1, x2):
        if (x1 - self.x1_center)**2 + (x2 - self.x2_center)**2 <= self.radius**2:
            return True
        return False
#создадим датафрейм, решающий поставленную задачу
circle = Circle(1, 2, 2.5)
df = objects_df.apply(lambda x: circle.is_in_circle(*x) ,axis = 1)
df[df==True]
print (df)
# ИЛИ тот же пример в другим вариантом решения (max-метрики)
# перевод на новую строку - к концу выводимой строки прикрепляется символ '\n'
abcde = [(4, 2), (3, 2), (1, -1), (-1, 1), (0,4)]
[print('abcde'[abcde.index(i)], end = ' \n') for i in list(filter(lambda x: abs(x[0]-1)<= 2.5 and abs(x[1]-2)<=2.5, abcde))]

# Планируется разбить объекты на 2 кластера На первой итерации работы алгоритма k-means были выбраны точки (2,3) и (1,1).
# После первой итерации алгоритма к кластеру, определяемому первой точкой, будут отнесены объекты
# (используется метрика Манхэттен)
import numpy as np
def dist(A,B):
    return np.abs((A-B)).sum()
A = np.array([4,  2])
B = np.array([3, 2])
C = np.array([1, -1])
D = np.array([-1, 1])
E = np.array([0, 4])
Center1 = np.array([2, 3])
Center2 = np.array([1, 1])
ABCDE = [A, B, C, D, E]
for vector in ABCDE:
        if dist(Center1, vector) < dist(Center2, vector):
            print('Первый кластер')
        else:
            print('Второй кластер')
# ИЛИ тот же пример в другим вариантом решения (метрика Манхэттен)
df = pd.DataFrame(np.array([[4,2],[3,2],[1,-1],[-1,1],[0,4]]),
                   columns=['P1', 'P2'], index=['A', 'B', 'C', 'D', 'E'])
df.index[print((df.P1 - 2).abs() + (df.P2 - 3).abs()) < ((df.P1 - 1).abs() + (df.P2 - 1).abs())]
# ИЛИ тот же пример в другим вариантом решения (метрика Манхэттен)
from scipy.spatial import distance
arr = {"a":[4,2],"b":[3,2],"c":[1,-1],"d":[-1,1],"e": [0,4]}
k1,k2 = [2,3],[1,1]
for i in arr.keys():
    if distance.cityblock(arr.get(i),k1)< distance.cityblock(arr.get(i),k2):
        print(i)


# *** Задача предсказания, линейная регрессия ***  (предсказывать количественные признаки, то есть решать задачу регрессии)
# Дана таблица, содержащая истинное значение целевого признака Y и предсказанное значение Y'.
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
y = np.array([1,2,3,4,5,-1,-2,-3,-4,-5])
y_pred = np.array([0,2,2,5,3,-1,-1,-4,-6,-5])
def mape(y, y_pred):
    return np.mean(np.abs((y - y_pred) / y)) * 100
print("Значение MAE равно: ", mae(y, y_pred))
print("Значение MAPE равно: ",mape(y, y_pred))

# Объекты в следующей таблице имеют один нецелевой признак Х.
# Где в таблице колонки X (0,1,2,3), Y (0,1,0,3)
# Построим модель линейно регрессии для предсказания признака Y.
from sklearn import linear_model as lm
# Create linear regression object
lr = lm.LinearRegression()
lr.fit([[0],[1],[2],[3]], [0,1,0,3])
print("(Intercept): X=", lr.intercept_)
print("(Coefficient): Y=", lr.coef_)

# Объекты в следующей таблице имеют два нецелевых признака Х1, Х2.
# Где в таблице колонки X1 (0,1,2,3), X2 (3,2,1,0), Y (0,1,0,3)
# Проблема в том, что существует сильная (даже линейная) зависимость между признаками X1,X2.
# Построим модель линейной регрессии с регуляризацией для предсказания
# признака Y (значение константы регуляризации С положить равным 1).
from sklearn.linear_model import Ridge
import numpy as np
alpha=1.0
X = np.array( [ [0,3], [1,2], [2,1], [3,0] ] )
y = np.array( [ 0, 1, 0, 3 ] )
#expand X by 1.0 column
X = np.insert(X, 0, values=1, axis=1 )
model = Ridge(alpha=alpha, fit_intercept=False )
model.fit(X, y )
print("coef:", model.coef_)
print("Полученная модель будет иметь вид: Y=0.62, X1+-0.10, X2+0.17")


# *** Классификация, kNN, кросс-валидация ***  (классификация, т.е. предсказывать значения номинальных признаков)
# Дана матрица ошибок, построенная по результатам работы некоторого алгоритма классификации.
TN = 25; FN = 20
FP = 10; TP = 15
Acc = (TN + TP) / (TN + TP + FN + FP)
Preс = TP / (TP + FP)
Rec = TP / (TP + FN)
# Общая точность (Accuracy), точность(Precision), полнота (Recall)
print('Accuracy: {0}\nPrecision: {1}\nRecall: {2}'.format("%.2f"%Acc, Preс, "%.2f"%Rec)) # "%.2f"% округление значения до 2х знаков после запятой

# Один очень тупой классификатор С относит все объекты к классу 1.
# Допустим,что выборка состоит из 50 объектов: 20 из них действительно принадлежат классу 0,
# а 30 из них действительно принадлежат классу 1.
TN = 0; FN = 0
FP = 20; TP = 30
Acc = (TN + TP) / (TN + TP + FN + FP)
Preс = TP / (TP + FP)
Rec = TP / (TP + FN)
print('Общая точность (Accuracy): {0}\nТочность(Precision): {1}\nПолнота (Recall): {2}'.format(Acc, Preс, "%.0f"%Rec))

# По тренировочной выборке из 70 элементов был построен некоторый классификатор.
# Мы взяли и проверили его качество на тестовой выборке, состоящей из 30 элементов.
# Сумма чисел TP+FP+FN+TN из матрицы ошибок равна?
TN = 0; FN = 0
FP = 0; TP = 30
# P = TP + FN – число истинных результатов,
# N = TN + FP – число ложных результатов.
# P + N - число элементов тестовой выборки
S = (TP+FP+FN+TN)
print('Сумма чисел TP+FP+FN+TN =: {0}\n'.format(S))

# Даны объекты тренировочной выборки. На вход алгоритму kNN подается объект F с признаками X1=0, X2=0.
# Где в таблице колонки X1 (-1,1,2,-3,3), X2 (0,2,-2,-1,2), Y (1,0,0,1,1)
# А строки 'A','B','C','D','E'
from sklearn.neighbors import KNeighborsClassifier
data = pd.DataFrame({'x1':[-1,1,2,-3,3], 'x2':[0,2,-2,-1,2], 'y':[1,0,0,1,1]},index=['A','B','C','D','E'])
F = pd.DataFrame({'x1':[0],'x2':[0]}, index=['F'])
X = data.drop(['y'], axis=1)
y = data.y
neighbors = [3, 5]
for neighbor in neighbors:
    clf = KNeighborsClassifier(n_neighbors=neighbor, metric='euclidean')
    clf.fit(X, y)
    print('For k={} class of F is: '.format(neighbor) + str(clf.predict(F)))
# ИЛИ такой вариант решения
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
X = np.array([[-1, 0], [1, 2], [2, -2], [-3, -1], [3, 2]])
Y = np.array([1, 0, 0, 1, 1])
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, Y)
print("При k=3 объект F будет отнесен к классу: ", model.predict([[0, 0]]))
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, Y)
print("При k=5 объект F будет отнесен к классу: ", model.predict([[0, 0]]))


# *** Деревья в машинном обуч *** (один из самых наглядных методов классификации и регрессии - деревья)
# Дана таблица, объекты в которой обладают нецелевыми признаками P1, P2, P3 и целевым признаком Y
# Где в таблице колонки p1 (0,0,1,1,0), p2 (1,1,1,1,1), p3 (0,0,0,1,1), Y (1,1,1,0,0)
# А строки 'A','B','C','D','E'
# Как известно в качестве условия ветвления выбирается признак с минимальным значением неопределенности Джини.
# Вычислим эту величину для каждого нецелевого признака.
from functools import partial
def Gini(x, y):
    def Pr(px, py=None):
        if py is None:
            return x.count(px) / len(x)
        lv = [y[i] for i in range(len(x)) if x[i] == px]
        if len(lv) != 0:
            return lv.count(py) / len(lv)
        else:
            return 0
    return Pr(0) * Pr(0, 0) * Pr(0, 1) + Pr(1) * Pr(1, 0) * Pr(1, 1)
p1 = [0, 0, 1, 1, 0]
p2 = [1, 1, 1, 1, 1]
p3 = [0, 0, 0, 1, 1]
y = [1, 1, 1, 0, 0]
G = partial(Gini, y=y)
for x, p in zip([G(p1), G(p2), G(p3)], ['p1', 'p2', 'p3']):
    print('Неопределенность Джини признака {} = {:.2f}'.format(p, x))


# *** Линейные классификаторы (ЛК) *** (на них строятся такие мощные модели МО как нейронные сети)
# Дана функция f(u,v)=u2-2v2-2u+4v+1
# Ее градиент в точке (0,0) равен (?,?)
# Будем искать точку минимума с помощью метода градиентного спуска. Длину шага полагаем равной 0,5.
# В какую точку мы попадем из точки (0,0) после одной итерации метода градиентного спуска? Ответ: (?,?)
from sympy import *
def gradient_sym(L, var_lst):
    '''
    L функция в символьном виде
    var_lst символьный список переменных
    '''
    d_lst=[]
    # находим градиент
    for var in var_lst:
        d_lst.append(diff(L, var))
    return d_lst
def gradient_down(grd_lst,start,step,count, var_lst):
    '''
    grd_lst символьный градиент
    var_lst символьный список переменных
    start список с кооррдинатами начала спуска
    step шаг спуска
    count количество шагов
    '''
    for c in range(1, count+1):
        # готовим координаты старта для градиента
        t_lst = []
        for i in range(0,len(var_lst)):
            t_lst.append((var_lst[i],start[i]))
        # делаем один шаг по антиградиенту
        gr_v = []
        for g in grd_lst:
            gr_v.append(g.subs(t_lst))
        #  вх градиент и конечная точка
        print('Gradient', gr_v)
        start = list(np.array(start) + np.array(gr_v)*(-1)* step)
        print ('Step :',c, start)
# Дана функция f(u,v)=u2-2v2-2u+4v+1
u, v = symbols("u v")
f = u**2 - 2*v**2 - 2*u + 4*v +1
# Ее градиент в точке (0,0). Длину шага полагаем равной 0,5.
f_gr = gradient_sym(f,[u,v])
gradient_down(f_gr,[0,0], 0.5,1,[u,v])

# Чему равен градиент функции f(u,v)=u*v+1 в точке (1,2)?
from sympy import symbols, diff
u, v = symbols('u v')
L = u * v + 1
coord = {v: 2, u: 1}
f1 = diff(L, u).subs(coord)
f2 = diff(L, v).subs(coord)
print('Градиент функции f(u,v)=u*v+1 в точке (1,2) равен: (' + str(f1) + ', ' + str(f2) + ')')

# ИЛИ еще так, градиент функции f(u,v)=u*v+1 в точке (1,2)?
from sympy import *
def gradient_sym(L, var_lst):
    '''
    L функция в символьном виде
    var_lst символьный список переменных
    '''
    d_lst=[]
    # находим градиент
    for var in var_lst:
        d_lst.append(diff(L, var))
    return d_lst
def gradient_down(grd_lst,start,step,count, var_lst):
    '''
    grd_lst символьный градиент
    var_lst символьный список переменных
    start список с кооррдинатами начала спуска
    step шаг спуска
    count количество шагов
    '''
    for c in range(1, count+1):
        # готовим координаты старта для градиента
        t_lst = []
        for i in range(0,len(var_lst)):
            t_lst.append((var_lst[i],start[i]))
        # делаем один шаг по антиградиенту
        gr_v = []
        for g in grd_lst:
            gr_v.append(g.subs(t_lst))
        #  вх градиент и конечная точка
        print('Gradient', gr_v)
        start = list(np.array(start) + np.array(gr_v)*(-1)* step)
        print ('Step :',c, start)
# Дана функция f(u,v)=u*v+1
u, v = symbols("u v")
f = u * v +1
f_gr = gradient_sym(f,[u,v])
# Ее градиент в точке (1,2).
gradient_down(f_gr,[1,2], 0.5,1,[u,v])

# Искусственный нейрон имеет 2 входа с весами 1, 2 соответственно.
# Ко входному сигналу прибавляется число 1 и применяется сигмоидная функция.
# Чему будет равно значение на выходе из нейрона, если на первый вход было подано число 1,
# а на второй вход было подано число -1?
# f(z) = sigmoid(z) = exp(z) / (exp(z) + 1), z = W1 * X1 + W2 * X2 + 1
import numpy as np
x1 = 1
x2 = -1
f = x1 + 2 * x2 + 1
print(np.e ** (f) / (np.e ** (f) + 1))

# Дана таблица с объектами (каждый объект имеет 2 нецелевых признака и известную метку класса).
# Линейный классификатор имеет вид: если a*X1+b*X2+c>0, то объект принадлежит классу 1 (иначе объект принадлежит классу -1).
# Какими должны быть значения весов a,b,c, чтобы классификатор правильно классифицировал все объекты в таблице?
# fill structures with given data
a = [1, 1, 1, 0, 0, 1]
b = [1, -1, 1, 1, 1, 0]
c = [1, 0, -1, 0, 1, 0]
# структура таблицы, где колонки X1,X2,Y
x1 = [-1, 3, 1, -1]
x2 = [3, -1, -1, 1]
y = [True, True, False, False]
def svm(a, b, c, x1, x2):
    return a * x1 + b *x2 + c > 0
for i in range(0, len(a)):
    success_variant = True
    for j in range(0, len(y)):
        if svm(a[i], b[i], c[i], x1[j], x2[j]) != y[j]:
            success_variant = False
            break
    if success_variant == True:
        print("{0} variant is successful, a = {1}, b = {2}, c = {3}".format(i, a[i], b[i], c[i]))
        break

# Построим линейный классификатор, в котором
# выражение [Mi<0] мажорируется функцией (1-M)^2 (^2 - это возведение  в квадрат).
# Получим следующее правило классификации:
from sympy import *
w1, w0 = symbols('w1 w0')
f = (1-2*w1+w0)**2 + (1-w1+w0)**2 + (1-w0)**2 + (1-w1-w0)**2 + (1-2*w1-w0)**2
print(solve((diff(f, w1), diff(f, w0)), (w1, w0)))


# *** Вероятностные алгоритмы ***
# Для объектов из таблицы с помощью некоторого алгоритма были получены вероятности принадлежности классу 1.
# (Колонка табл 'class_1':[0.6,0.81,0.5,0.9,0.7,0.75])
# Истинные метки классов объектов также известны. (Колонка табл 'true':[0,1,0,1,0,1])
# Площадь под ROC-кривой (величина AUC) равна?
import pandas as pd # pip install pandas
data_roc = pd.DataFrame({'class_1':[0.6,0.81,0.5,0.9,0.7,0.75],'true':[0,1,0,1,0,1]})
print(data_roc)
from sklearn.metrics import roc_auc_score
y_true = data_roc.drop('true', axis=1)
y_scores = data_roc.true
roc_auc_score(y_scores, y_true)
print('Площадь под ROC-кривой (величина AUC) равна: ', roc_auc_score(y_scores, y_true))

# Для объектов из таблицы с помощью некоторого алгоритма были получены вероятности принадлежности классу 1.
# Истинные метки классов объектов также известны.
import pandas as pd
data_roc = pd.DataFrame({'class_1':[0.6,0.81,0.5,0.9,0.7,0.75],'true':[1,1,0,1,0,0]})
print(data_roc)
from sklearn.metrics import roc_auc_score
y_true = data_roc.drop('true', axis=1)
y_scores = data_roc.true
roc_auc_score(y_scores, y_true)
print('Площадь под ROC-кривой (величина AUC) равна: ', "%.2f"%roc_auc_score(y_scores, y_true))
# ИЛИ так, еще вариант решения
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
example5 = pd.DataFrame({'Probability':[0.6,0.81,0.5,0.9,0.7,0.75], 'Value':[1,1,0,1,0,0]}, index = ['A','B','C','D','E','F'])
tpr, fpr, thresholds = roc_curve(example5.Value, example5.Probability)
plt.figure()
sns.set(rc = {'figure.figsize':(10,6)})
plt.plot(tpr, fpr, label = 'AUC score = {}'.format("%.2f"%auc(tpr,fpr)), lw=2)
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()


# *** Ансамбли алгоритмов ***
# *** Мотивация и простые способы отбора признаков ***
# Рассмотрим работу SMOTE-алгоритма по синтезу новых объектов ( Формула f = aA+(1-a)B )
# Даны два объекта A и B (строки), где колонки p1 (1,5), p2 (2,10), Y (1,1)
# Параметр линейной комбинации а равен 0,25.
A1 = 1
B1 = 5
A2 = 2
B2 = 10
a = 0.25
p1 = a*A1+(1-a)*B1
p2 = a*A2+(1-a)*B2
print('Значения признаков синтетического объекта равны: Р1 =' ,p1, ', Р2 =', p2)

# В таблице дана точность модели построенной с помощью нецелевых признаков U, V, W, X
# (если в таблице стоит 0, то соответствующий признак не использовался при построении модели).
# Используя жадный алгоритм отбора признаков (который начинает свою работу с полного набора признаков {U,V,W,X}),
# найдем все признаки, попавшие в оптимальный набор.
import pandas as pd
data=pd.DataFrame({'U':[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],'V':[0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
                   'W':[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],'X':[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
                   'Y':[0,0.75,0.55,0.65,0.5,0.82,0.87,0.62,0.8,0.84,0.85,0.7,0.81,0.8,0.75,0.6]})
data['S']=data.iloc[:,0:4].sum(axis=1)
data.sort_values(by='S')
print(data.sort_values(by='S'))

