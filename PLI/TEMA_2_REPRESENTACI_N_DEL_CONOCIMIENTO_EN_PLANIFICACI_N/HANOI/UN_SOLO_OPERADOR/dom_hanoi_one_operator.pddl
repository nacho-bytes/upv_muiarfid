;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; HANOI
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain hanoi)
  (:requirements :strips :typing)
  (:types disk peg)
  (:predicates (at ?x - disk ?y - (either disk peg))
               (clear ?x - (either disk peg))
               (smaller ?x - disk ?y - (either disk peg)))


  (:action MOVE-DISK
	:parameters (?disk - disk ?from - (either disk peg) ?new-below - (either disk peg))
	:precondition (and (at ?disk ?from) (clear ?disk)(clear ?new-below)
				  (smaller ?disk ?new-below))
      :effect (and (at ?disk ?new-below) (clear ?from)
                   (not (clear ?new-below)) (not (at ?disk ?from))))
)



 