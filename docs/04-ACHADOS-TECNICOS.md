# Cinco achados que valem além deste projeto

Coisas descobertas no caminho que mudam como medir, não só o que medimos.

---

## 1. O caminho da curvatura não é reprodutível na GPU

Descoberto retreinando os 37 checkpoints perdidos. Mesma seed, mesmo código,
mesmo split, mesma máquina:

| | \|Δ\| médio no F-borda | máximo | melhor época igual |
|---|---|---|---|
| **B0, sem termo geométrico** | **0,0000** | 0,0000 | **6/6** |
| **B1 e B3, com curvatura métrica** | **0,0148** | **0,0389** | **0/7** |

O berHu puro reproduz exato, até a época de early stop. Todo braço com curvatura
não reproduz nenhuma métrica.

**A causa:** `F.pad(mode="replicate")` no `geometry.py` (linhas 67, 75, 91, 92),
dentro do `surface_curvatures`, que é chamado sobre o `pred` e portanto tem
backward. O backward do replication padding na CUDA usa `atomicAdd` e é **não
determinístico**, caso documentado do PyTorch. O `metrics.py` também usa
replicate, mas sob `@torch.no_grad()`, então não contamina.

**A correção:** trocar por fatiamento e `torch.cat`, que dá o mesmo forward e um
backward determinístico.

**Por que importa:** o ruído de reexecução é **da mesma ordem do efeito** que
reportamos (-0,019 a -0,025). Qualquer ablação que ordene configurações com uma
seed cada está ordenando ruído. Quantas seeds para distinguir 0,02:

| cenário | desvio | seeds por config |
|---|---|---|
| como está | 0,018 | **13** |
| com a correção | 0,012 | **6** |

**O lado bom:** como o retreino não reproduziu, ele virou replicação
independente. Isso dobrou o n de cada braço, de 6 para 11 ou 12, e a conclusão
ficou **mais forte**: de 4 vitórias em 30 para 3 em 47.

---

## 2. O `fmax` não se mexe com nada

O `boundary_fscore` usa um limiar fixo relativo ao percentil 99 do gradiente de
cada imagem. O `boundary_fmax` é o F-score no **melhor limiar de cada modelo**.

| | F-borda (limiar fixo) | fmax (melhor limiar) |
|---|---|---|
| zero-shot | 0,5402 | 0,7674 |
| B0 controle | 0,5954 (**+0,0540**) | 0,7791 (**+0,0117**) |
| B3 teto 50 | 0,5765 | 0,7713 |

O ganho aparente de +0,054 cai para +0,0117 quando cada modelo usa o próprio
melhor limiar. Ou seja, **o que o treino consegue é recalibrar a distribuição de
gradiente** para que o limiar fixo caia num lugar melhor. O mapa de bordas em si
quase não melhora.

E não é falta de resolução: a 768 px o `fmax` **piora** (-0,0344, pior em 3 de 3
seeds) enquanto RMSE e delta1 melhoram em 3 de 3. Seis configurações de perda e
uma mudança de resolução, e o teto de ~0,77 não sai do lugar.

**Consequência:** ao reportar ganho de borda, sempre mostrar `fmax` e `f_auc` ao
lado do `fscore`. Sozinho, o `fscore` de limiar fixo confunde calibração com
qualidade.

---

## 3. O n efetivo do teste é 13 cenas, não 485 imagens

Quadros consecutivos da mesma sequência são quase o mesmo dado. Medindo no
zero-shot, cena a cena:

- desvio entre cenas: **0,2147**
- erro padrão da média com n=13: **0,0595**
- maior diferença entre braços: **~0,06**

O ruído entre cenas é da mesma ordem do efeito inteiro. Isso não invalida a
comparação entre braços, que é pareada e usa as mesmas cenas dos dois lados, mas
invalida qualquer afirmação sobre o **valor absoluto** e sobre generalização além
dessas 13 cenas.

É a mesma lição que a linha do bokeh aprendeu com a mesa LF-repro, que tinha 500
imagens e 10 cenas.

---

## 4. Duas cenas dominam o conjunto de teste

| cena | n | F-borda | AbsRel | d1 |
|---|---|---|---|---|
| **seq0020** | 77 | 0,179 | 0,763 | **0,050** |
| **seq0043** | 23 | 0,458 | **2,279** | 0,368 |
| mediana das outras | | ~0,63 | ~0,10 | ~0,90 |

`d1 = 0,05` significa que 5% dos pixels estão dentro de 25% do GT. Excluindo as
duas (21% do teste, 2 de 13 cenas), o AbsRel do zero-shot cai de **0,3591 para
0,1637** e o F-borda sobe de 0,5341 para 0,6097.

Investigado e **descartado**: não é artefato de métrica (um teste controlado com
predição sintética de erro relativo uniforme recupera ~0,08 de AbsRel nas duas) e
não é o espaço do alinhamento afim (alinhar em profundidade ou em disparidade dá
diferença marginal, exceto na seq0043, onde vale 2x).

A causa da `seq0020` continua sem identificação e precisa de inspeção visual.

Um dado relacionado: a correlação entre a **faixa de profundidade da cena**
(p95/p5) e o AbsRel é **+0,906**. O DepthPro erra mais em cenas de grande
amplitude, e isso é real, não artefato.

---

## 5. A máscara de validade muda o F-score de borda

O dataset zera os pixels inválidos, e a fronteira entre região válida e os zeros
vira uma **borda falsa no alvo** que a predição, sendo contínua, não tem. O
efeito é precisão alta e recall derrubado.

Medido: uma predição quase perfeita cai de bF-max **1,000 para 0,792 com apenas
1%** da imagem zerada.

No Spring isso pesa: **seq0045 tem 46% mascarado, seq0020 21%, seq0014 17%**.

A correção está no `origin/main` do repositório original, e consiste em erodir a
validade pelo raio da tolerância e descartar bordas que tocam região inválida.
**A branch `fix-geometry` não tem essa correção**, e foi dela que a ablação
partiu, o que explica o zero-shot dar 0,4861 lá e 0,5402 aqui, sendo o mesmo
modelo sem fine-tune.

---

## Bônus: duas coisas de infraestrutura que custaram caro

**Guarda de memória protege o seu job, não o do vizinho.** Checar memória livre
antes de lançar impede que o seu treino tome OOM, mas não impede que ele cause
OOM em quem está ciclando itens na mesma placa. Foi assim que dois itens de uma
fila de avaliação de bokeh morreram: a guarda viu 78 GB livres num intervalo
entre itens e entrou.

**Subir para o Hub no fim da faixa deixa uma janela grande demais.** A primeira
fila de retreino só publicava depois de 3 seeds, deixando peso horas só no
`/raid`. A fila seguinte publica a cada seed que fecha. A diferença é a janela em
que a regra "checkpoint que não está no Hub não existe" fica violada.
