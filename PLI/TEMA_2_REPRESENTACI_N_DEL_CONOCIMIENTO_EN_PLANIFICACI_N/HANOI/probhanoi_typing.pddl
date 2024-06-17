(define (problem probhanoi1)
(:domain hanoi)
(:objects L M S - disk 
          P1 P2 P3 - peg)

(:init (at S M)(at M L)(at L P1)(clear S)(clear P2)(clear P3))

(:goal (and (at S M)(at M L)(at L P3)))
)