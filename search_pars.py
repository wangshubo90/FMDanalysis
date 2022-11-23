from fitVessel import *
from sklearn.metrics import jaccard_score

def Dice(gt, pred):
    j = jaccard_score(gt.flatten()>0.5, pred.flatten()>0.5)
    return 2*j / (j+1)

def test(image_list, label_list):
    
    results = []
    for img, label in zip(image_list, label_list):
        guess, x_grid, image, edge1, edge2 = fit(file, plot=True, init_func=polynomial_init_condition, edge_func=calc_edge_polynomial, fit_func=fit_vessel, refimg=gimage)
        