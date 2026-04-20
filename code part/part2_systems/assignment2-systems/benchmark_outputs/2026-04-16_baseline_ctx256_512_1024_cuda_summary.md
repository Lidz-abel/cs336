## context_length=256

| size | forward mean (ms) | forward std (ms) | forward+backward mean (ms) | forward+backward std (ms) | status |
|---|---:|---:|---:|---:|---|
| small | 26.219 | 0.133 | 81.132 | 0.130 | PASS |
| medium | 78.323 | 0.052 | 235.239 | 0.908 | PASS |
| large | 160.916 | 1.282 | 474.260 | 0.725 | PASS |
| xl | 319.407 | 1.207 | 930.446 | 1.806 | PASS |
| 2.7b | 474.254 | 1.349 | 1288.198 | 1.971 | PASS |

## context_length=512

| size | forward mean (ms) | forward std (ms) | forward+backward mean (ms) | forward+backward std (ms) | status |
|---|---:|---:|---:|---:|---|
| small | 62.471 | 0.167 | 195.324 | 0.212 | PASS |
| medium | 185.372 | 0.154 | 564.280 | 0.519 | PASS |
| large | 371.629 | 0.778 | 1112.590 | 1.037 | PASS |
| xl | 719.213 | 0.566 | 2104.343 | 1.612 | PASS |
| 2.7b | 1025.964 | 0.856 | - | - | forward=PASS, forward_backward=OOM |

## context_length=1024

| size | forward mean (ms) | forward std (ms) | forward+backward mean (ms) | forward+backward std (ms) | status |
|---|---:|---:|---:|---:|---|
| small | 168.471 | 0.119 | 535.220 | 0.165 | PASS |
| medium | 484.431 | 0.406 | 1507.564 | 0.893 | PASS |
| large | 986.346 | 0.655 | - | - | forward=PASS, forward_backward=OOM |
| xl | 1797.081 | 1.761 | - | - | forward=PASS, forward_backward=OOM |
| 2.7b | 2336.771 | 2.753 | - | - | forward=PASS, forward_backward=OOM |
