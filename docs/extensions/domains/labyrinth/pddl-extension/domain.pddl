(define (domain labyrinth)
    (:requirements :typing :arrays :bounded-integers)

    (:types
        card direction - object
        idx            - (number 0 3)
        grid           - (array 4 4 card)
    )

    (:constants
        N S E W - direction
    )

    (:predicates
        (connections ?c - card ?d - direction)
    )

    (:functions
        (card_at) - grid
        (robot_at) - card
    )

    (:action move_north
        :parameters (?r ?c - idx)
        :precondition (and
            (= (robot_at) (read (card_at) ?r ?c))
            (connections (read (card_at) ?r ?c) N)
            (connections (read (card_at) (- ?r 1) ?c) S)
        )
        :effect (and
            (write (robot_at) (read (card_at) (- ?r 1) ?c))
        )
    )

    (:action move_south
        :parameters (?r ?c - idx)
        :precondition (and
            (= (robot_at) (read (card_at) ?r ?c))
            (connections (read (card_at) ?r ?c) S)
            (connections (read (card_at) (+ ?r 1) ?c) N)
        )
        :effect (and
            (write (robot_at) (read (card_at) (+ ?r 1) ?c))
        )
    )

    (:action move_east
        :parameters (?r ?c - idx)
        :precondition (and
            (= (robot_at) (read (card_at) ?r ?c))
            (connections (read (card_at) ?r ?c) E)
            (connections (read (card_at) ?r (+ ?c 1)) W)
        )
        :effect (and
            (write (robot_at) (read (card_at) ?r (+ ?c 1)))
        )
    )

    (:action move_west
        :parameters (?r ?c - idx)
        :precondition (and
            (= (robot_at) (read (card_at) ?r ?c))
            (connections (read (card_at) ?r ?c) W)
            (connections (read (card_at) ?r (- ?c 1)) E)
        )
        :effect (and
            (write (robot_at) (read (card_at) ?r (- ?c 1)))
        )
    )

    (:action rotate_col_up
        :parameters (?c - idx)
        :precondition (and
            (forall (?r - idx)
                (not
                    (= (robot_at) (read (card_at) ?r ?c))
                )
            )
        )
        :effect (and
            (forall (?r - idx)
                (when (>= ?r 1)
                    (write ((card_at) (- ?r 1) ?c) (read (card_at) ?r ?c))
                )
            )
            (write ((card_at) 3 ?c) (read (card_at) 0 ?c))
        )
    )

    (:action rotate_col_down
        :parameters (?c - idx)
        :precondition (and
            (forall (?r - idx)
                (not
                    (= (robot_at) (read (card_at) ?r ?c))
                )
            )
        )
        :effect (and
            (forall (?r - idx)
                (when (>= ?r 1)
                    (write ((card_at) ?r ?c) (read (card_at) (- ?r 1) ?c))
                )
            )
            (write ((card_at) 0 ?c) (read (card_at) 3 ?c))
        )
    )

    (:action rotate_row_left
        :parameters (?r - idx)
        :precondition (and
            (forall (?c - idx)
                (not
                    (= (robot_at) (read (card_at) ?r ?c))
                )
            )
        )
        :effect (and
            (forall (?c - idx)
                (when (>= ?c 1)
                    (write ((card_at) ?r (- ?c 1)) (read (card_at) ?r ?c))
                )
            )
            (write ((card_at) ?r 3) (read (card_at) ?r 0))
        )
    )

    (:action rotate_row_right
        :parameters (?r - idx)
        :precondition (and
            (forall (?c - idx)
                (not
                    (= (robot_at) (read (card_at) ?r ?c))
                )
            )
        )
        :effect (and
            (forall (?c - idx)
                (when (>= ?c 1)
                    (write ((card_at) ?r ?c) (read (card_at) ?r (- ?c 1)))
                )
            )
            (write ((card_at) ?r 0) (read (card_at) ?r 3))
        )
    )
)
