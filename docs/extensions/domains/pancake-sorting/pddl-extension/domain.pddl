(define (domain pancake)

    (:requirements :typing :arrays :bounded-integers)

    (:types
        jo tu ell - object
        pancakes - (number 0 4)
        stack    - (array 5 pancakes)
    )

    (:functions
        (pancake_stack) - stack
    )

    (:action flip
        :parameters (?f - pancakes)
        :precondition ()
        :effect (and
            (forall (?i - pancakes)
                (when (<= ?i ?f)
                    (write
                        (pancake_stack) (?i)
                        (read (pancake_stack) (- ?f ?i))
                    )
                )
            )
        )
    )
)
