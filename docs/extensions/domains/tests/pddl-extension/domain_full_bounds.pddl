(define (domain pancake-int)

    (:requirements :typing :fluents :bounded-integers)

    (:types
        idx - (number 0 9)
    )

    (:functions
        (val ?i - idx) - idx
    )

    ;; Phase 1: copy reversed prefix into tmp
    (:action flip
        :parameters (?f - idx)
        :effect (and
            (forall (?i - (number 0 ?f))
                (assign (val ?i) (val (- ?f ?i)))
            )
        )
    )
)
