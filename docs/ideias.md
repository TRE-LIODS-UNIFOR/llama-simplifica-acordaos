Esse documento guardará os possíveis caminhos a serem seguidos com o desenvolvimento deste projeto. Ideias diferentes serão separadas por uma linha horizontal.

---

### Data

11/09

### Situação:

O modelo não está reconhecendo os termos que deve simplificar.

### Solução:

-   Passar junto ao acórdão um glossário de termos jurídicos, indicando que a LLM deve substituir os termos do documento original pelos do glossário sempre que possível.

---

### Data

13/09

### Situação:

O modelo não está reconhecendo os termos que deve simplificar. As respostas são inconsistentes, mudando muito de uma para a outra.

### Solução:

Em chat:

<center>

```mermaid
    flowchart
        A["Passar documento ao LLaMa."] --> B["LLaMa produz uma versão reduzida."]
        B --> C["Passar um glossário de termos jurídicos ao LLaMa."]
        C --> D["Pedir ao LLaMa que substitua no seu texto os termos encontrados no glossário, reescrevendo as frases sempre que necessário."]
```

</center>
<br>

Como fazer:
1.  Usar modo de QA (Questions and Answers);
2.  Enviar a instrução de redução do documento;
3.  Enviar a instrução de simplificação junto ao glossário. Usar um fluxograma ou outra forma de especificar a lógica que deve ser empregada;
4.  Coletar resposta.

<center>

![](etapas.jpg)

</center>
