;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; HANOI
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain hanoi)
  (:requirements :strips :typing :equality)
  (:types disk peg)
  (:predicates (at ?x - disk ?y - (either disk peg))
               (clear ?x - (either disk peg)))

  (:action move-large
	     :parameters (?x - peg ?y - peg)
	     :precondition (and (at L ?x) (clear L)(clear ?y))
           :effect
	               (and (not (at L ?x))(at L ?y)(not (clear ?y))(clear ?x)))



 (:action move-medium
	     :parameters (?x - (either peg disk) ?y - (either disk peg))
	     :precondition (and (at M ?x)(clear M)(clear ?y)
                              (not (= ?y S)))
    	     :effect
	     (and (not (at M ?x))(at M ?y)(not (clear ?y))(clear ?x)))



 (:action move-small
	     :parameters (?x - (either peg disk) ?y - (either disk peg))
	     :precondition (and (at S ?x)(clear S)(clear ?y))
	     :effect
	     (and (not (at S ?x))(at S ?y)(not (clear ?y))(clear ?x)))
)



 