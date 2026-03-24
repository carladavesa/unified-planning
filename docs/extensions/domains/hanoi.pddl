
;; The Towers of Hanoi problem (formalisation by Hector Geffner).

;(define (domain hanoi)
;  (:requirements :strips)
;  (:predicates (clear ?x) (on ?x ?y) (smaller ?x ?y))
;
;  (:action move
;    :parameters (?disc ?from ?to)
;    :precondition (and (smaller ?to ?disc) (on ?disc ?from)
;		       (clear ?disc) (clear ?to))
;    :effect  (and (clear ?from) (on ?disc ?to) (not (on ?disc ?from))
;		  (not (clear ?to))))
;  )

(define (domain hanoi)
    (:requirements :adl :fluents :numeric-fluents :typing)

    (:types
        peg disc - object
        this - peg
        range - (number 0 5)
        stack - (array 5 range)
    )

    (:predicates
        (on ?x - disc ?y - object)
        (smaller ?x - disc ?y - object)
    )

    (:functions
        (tower ?p - peg) - stack
        (top ?p - peg) - range
    )


    (:action move
        :parameters (?d - disc ?from - object ?to - object)
        :precondition (and
            (smaller ?d ?to)
            (on ?d ?from)
            (forall (?w - disc)
            (and (not (on ?w ?d))
            (not (on ?w ?to))))
        )
        :effect (and
            (on ?d ?to)
            (not (on ?d ?from))
        )
    )
)