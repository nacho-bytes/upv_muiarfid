(define (problem probhanoi1)
(:domain hanoi)
(:objects L M S - disk 
          P1 P2 P3 - peg)

(:init (at S M)(at M L)(at L P1)(clear S)(clear P2)(clear P3)(smaller S M)(smaller S L)(smaller M L)(smaller S P1)
        (smaller S P2)(smaller S P3)(smaller M P1)(smaller M P2)(smaller M P3)(smaller L P1)(smaller L P2)
         (smaller L P3))

(:goal (and (at S M)(at M L)(at L P3)))
)