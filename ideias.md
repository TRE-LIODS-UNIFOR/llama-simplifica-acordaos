Adicionar anotações na etapa de préprocessamento, como:

- Sempre que o texto mencionar o recorrente: "...o recorrente (Antônio Ilomar Cruz Vasconcelos) [...]"

23/11, 00:13

Sobre os problemas de desempenho:

* Vectorstore pode estar pesando bastante. Medir e determinar se valeria a pena hospedar em outra máquina.
* Verificar se resultados intermediários das etapas de processamento estão ocupando muita memória.
* Guardar resultados em disco para liberar memória, e apenas recuperá-los quando necessário for.
