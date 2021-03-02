# Results of Chapter 6 exercise

Given the results for recognizing the digit 5:

```
5 => Accuracy: 96.37%  Loss: 0.11896
```

I figured that the digit most likely to be recognised was a 1. So I tried that and some other digits to test my hypothesis.

```
1 => Accuracy: 99.03%  Loss: 0.0427
0 => Accuracy: 98.99%  Loss: 0.0482
3 => Accuracy: 96.98%  Loss: 0.1099
7 => Accuracy: 98.14%  Loss: 0.0652
```

Looks like I'm probably right. My hypothesis was that it's probably the simplest digit to draw and that the only digit likely to be confused with it is a 7 (hence why I tried that as well).
