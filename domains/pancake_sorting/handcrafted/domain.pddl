(define (domain pancake-sorting)
    (:requirements :typing)
    (:types tile position)
    (:predicates
        (at ?t - tile ?p - position)
        (lte ?p1 ?p2 - position)
        (flipat ?p1 ?p2 ?p)
    )

    (:functions
        (total-cost) - number
    )

    (:action flip
        :parameters (?pala - position)
        :precondition ()
        :effect (and
            (forall (?p ?pnext - position ?t ?tnext - tile)
                (when (and (lte ?p ?pala) (at ?t ?p) (flipat ?pala ?p ?pnext) (at ?tnext ?pnext))
                    (and (not (at ?t ?p)) (at ?tnext ?p)
                    )
                )
            )
            (increase (total-cost) 1)   
        )
    )
)