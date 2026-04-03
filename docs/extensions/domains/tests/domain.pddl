;;
;; Test domain for PDDL extension features:
;;   - bounded integers  (number lo hi)
;;   - arrays            (array N elem-type) / array.mk / read / write
;;
;; Note: sets (set elem-type / set.mk) parse correctly but SETS_REMOVING
;;       does not yet support integer-element sets, so they are omitted here.
;;

(define (domain test-features)

    (:requirements :typing :adl :fluents :numeric-fluents :arrays :bounded-integers)

    (:types
        player     - object

        ;; Bounded integer: scores in [0, 4]
        score      - (number 0 4)

        ;; Array type: a board holds 3 score slots
        board      - (array 3 score)
    )

    (:functions
        ;; Array fluent: each player has a 3-slot board of scores
        (slots ?p - player) - board

        ;; Numeric fluent (bounded integer): overall best score per player
        (best ?p - player) - score
    )


    ;; Write a value ?v into slot ?i of player ?p's board.
    ;; Precondition: the slot currently holds 0 (empty) and ?v is greater than 0.
    (:action fill-slot
        :parameters (?p - player ?i - score ?v - score)
        :precondition (and
            (> ?v 0)
            (= (read (slots ?p) (?i)) 0)
        )
        :effect (and
            (write (slots ?p) (?i) ?v)
        )
    )

    ;; Clear a slot: write 0 back to slot ?i of player ?p's board.
    ;; Precondition: the slot is currently non-zero.
    (:action clear-slot
        :parameters (?p - player ?i - score)
        :precondition (and
            (> (read (slots ?p) (?i)) 0)
        )
        :effect (and
            (write (slots ?p) (?i) 0)
        )
    )

    ;; Promote the score at slot ?i to be the player's new best,
    ;; if it is strictly greater than the current best.
    (:action promote-best
        :parameters (?p - player ?i - score)
        :precondition (and
            (> (read (slots ?p) ?i) (best ?p))
        )
        :effect (and
        (assign (best ?p) (read (slots ?p) (?i)))
        )
    )

)
