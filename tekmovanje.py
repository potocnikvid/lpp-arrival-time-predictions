from sys import argv
import numpy as np
import csv
import docker
from generiraj import lpputils as lpp
from generiraj import linear
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import scipy

class Tekmovanje:
  def __init__(self, train, test, prazniki, pocitnice, sneg):
    self.train = train
    self.test = test
    self.prazniki = prazniki
    self.pocitnice = pocitnice
    self.sneg = sneg

    self.drivers_unique = list(dict.fromkeys(train[:, 1]))
    self.routes = list(dict.fromkeys(train[:, 3]))

    self.months = [1, 2, 10, 11, 12]

    
    self.client = docker.from_env()
    self.client.containers.run("ubuntu", "echo hello world")



  def compute(self):
    x = self.day_drivers_week_snow_1(self.train)
    x_sp = scipy.sparse.csr_matrix(x)
    y = self.duration(self.train)
    lr = linear.LinearLearner(lambda_=1.)
    model = lr(x_sp, y)
    test_matrix = self.day_drivers_week_snow_1(self.test)
    test_prediction_duration = [model(l) for l in test_matrix]

    return test_prediction_duration

  def day_drivers_week_snow_1(self, data, interval=288):
    x = np.zeros((len(data), interval + 1440 + 3 + len(self.drivers_unique)))
    for i in range(len(data)):
      start = lpp.parsedate(data[i, 6])
      time = (start.hour * 60 + start.minute)*interval//1440
      day = 24 * (start.isoweekday() - 1) + start.hour
      date = data[i, 6][0:10]
      driver = data[i, 1]
      if start.month in self.months:
        x[i, time] = 1
        x[i, interval + day] = 1
        if date in self.prazniki:
          x[i, interval + 1440 + 1] = 1
        if date in self.pocitnice:
          x[i, interval + 1440 + 2] = 1
        if date in self.sneg:
          x[i, interval + 1440 + 3] = 1
        if driver in self.drivers_unique:
          x[i, interval + 1440 + 3 + self.drivers_unique.index(driver)] = 1        
    return x[~np.all(x == 0, axis=1)]

  def day_drivers_week_snow(self, data, interval=288):
    x = np.zeros((len(data), 7 + interval*7 + 3 + len(self.drivers_unique)))
    for i in range(len(data)):
      start = lpp.parsedate(data[i, 6])
      time = ((start.isoweekday() - 1) * 1440 + start.hour * 60 + start.minute)*interval//10080
      date = data[i, 6][0:10]
      driver = data[i, 1]
      if start.month in self.months: 
        x[i, start.isoweekday() - 1] = 1
        x[i, 7 + time] = 1
        if date in self.prazniki:
          x[i, 7 + interval*7 + 1] = 1
        if date in self.pocitnice:
          x[i, 7 + interval*7 + 2] = 1
        if date in self.sneg:
          x[i, 7 + interval*7 + 3] = 1
        if driver in self.drivers_unique:
          x[i, 7 + interval*7 + 3 + self.drivers_unique.index(driver)] = 1
    return x[~np.all(x == 0, axis=1)]
   
  def print_results(self, test, test_prediction_duration, filename):
    f = open(filename + ".txt", "wt")
    for len, l in zip(test_prediction_duration, test):
      start_time = l[6]
      end_time = lpp.tsadd(start_time, len)
      f.write(end_time + "\n")
    f.close()

  def duration(self, data):
    dur = []
    for e in data:
      if lpp.parsedate(e[6]).month in self.months:
        dur.append(lpp.tsdiff(e[8], e[6]))
    return np.array(dur)
    
  def read_file(self, file_path):
    f = open(file_path, 'r') 
    csvreader = csv.reader(f, delimiter='\t')
    headers = next(csvreader)
    list = [d for d in csvreader]
    return np.array(list)



def testing(test, prediction_duration, test_results_filename):
  test_results = read_file(test_results_filename)
  results_duration = np.array([lpp.tsdiff(result[0], line[6]) for line, result in zip(test, test_results)])
  print("MAE:", mean_absolute_error(results_duration, prediction_duration))


def print_results_routes(test, route_predictions, filename='results'):
  f = open(filename + ".txt", "wt")
  results = []
  for l in test:
    route = l[3]
    start_time = l[6]
    pred = route_predictions[route]
    len = route_predictions[route].pop(0)
    po = route_predictions[route]
    end_time = lpp.tsadd(start_time, len)
    results.append(len)
    f.write(end_time + "\n")
  f.close()
  return np.array(results)

def read_file(file_path):
  f = open(file_path, 'r')
  csvreader = csv.reader(f, delimiter='\t')
  _ = next(csvreader)
  list = [d for d in csvreader]
  return np.array(list)

if __name__ == "__main__":
  train = read_file("train.csv.gz")
  test = read_file("test.csv.gz")

  prazniki = read_file('prazniki.txt')
  pocitnice = read_file('pocitnice.txt')
  sneg = read_file('sneg.txt')

  routes = list(dict.fromkeys(train[:, 3]))

  route_predictions = {}
  for r in routes:
    if r in test[:, 3]:
      train_route = train[train[:, 3] == r]
      test_route = test[test[:, 3] == r]
      route_predictions[r] = Tekmovanje(train_route, test_route, prazniki, pocitnice, sneg).compute()

  results = print_results_routes(test, route_predictions, "results")
  # testing(test, results, 'self_results.csv.gz')

  