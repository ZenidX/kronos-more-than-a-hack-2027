# Plan de trabajo · Kronos

Entrega del Initial Challenge: **17-11-2026, 23:59 (CET)**. El formulario llega por correo el 14-11.

## 1. Imprescindible para entregar

- [ ] **Equipo**: todos los miembros registrados con el nombre «Kronos» (mínimo 3 para pasar de fase).
- [ ] **Formulario** (cuenta de Google): nombre del equipo, `Kronos.csv`, URL del repositorio, `Kronos.pdf`.
- [ ] **Validar el fichero exacto que se sube**: `python evaluation/validate_submission.py submission/Kronos.csv`.
- [ ] **Documento técnico en PDF** a partir de [`submission/Kronos.md`](../submission/Kronos.md), con las figuras.
- [ ] **Repositorio público** antes de enviar, revisando antes todo el historial: ningún dato del partido
      (`data/` está en `.gitignore`) ni nada derivado de él.

## 2. Mejoras del modelo (en orden de impacto esperado)

Regla para aceptar una mejora: **solo entra si mejora el spellF1 medio en los 10 partidos reservados de
SkillCorner** (`python -m kronos.tune`). Nunca se decide con los 5 minutos públicos del reto.

1. **Fases del Smart Tagging como cuarta pista.** Hoy solo se usan los saques iniciales. Las fases con
   equipo (`BUILD UP`, `PROGRESSION`, `DEFENDING IN …`) dicen quién tiene o no tiene el balón, justo en
   el 41 % del tiempo sin balón. Los partidos externos no traen Smart Tagging, así que su peso se fija
   con criterio y se documenta como tal.
2. **Saques de banda y de puerta.** Balón que desaparece cerca de una línea y jugadores que frenan →
   balón parado. Ajustable con SkillCorner.
3. **Control sostenido por el mismo jugador**, no solo "el más cercano de cada equipo": separa control
   de rebote.
4. **Más rasgos de forma de equipo**: presión sobre el último punto donde se vio el balón, compacidad,
   altura de la línea defensiva.
5. **Robustez**: intervalos bootstrap en los partidos reservados y repetir el ajuste con otra semilla.

## 3. Después del 17-11 (si pasamos al top 20)

- Vídeo del partido con el indicador de posesión superpuesto, para el pitch. Antes de entregar no se
  puede: la aclaración de la organización del 3-11 prohíbe comparar la salida con el vídeo.
- Demo y presentación para el Innovation Challenge.

## Normas que no se tocan

- Nada etiquetado a mano sobre el partido del reto, ni para entrenar ni para validar.
- Los 5 minutos públicos solo comprueban que el pipeline está bien; no eligen parámetros.
- Ningún dato humano sobre ese partido (eventos de Opta, StatsBomb, etc.).
- Nada de modelos comerciales que resuelvan la tarea; el algoritmo es nuestro.
- Los datos del partido no entran en el repositorio.
