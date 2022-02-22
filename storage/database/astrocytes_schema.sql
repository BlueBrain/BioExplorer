create table population
(
    guid        integer not null
        constraint population_pk
            primary key,
    name        varchar not null,
    description varchar
);

alter table population
    owner to bioexplorer;

create unique index population_guid_uindex
    on population (guid);

create unique index population_name_uindex
    on population (name);

create table node
(
    guid                 integer          not null
        constraint node_pk
            primary key,
    population_guid      integer          not null,
    x                    double precision not null,
    y                    double precision not null,
    z                    double precision not null,
    radius               double precision not null,
    model_template_guid  varchar          not null,
    model_type_guid      varchar          not null,
    morphology           varchar          not null,
    morphology_type_guid varchar          not null
);

alter table node
    owner to bioexplorer;

create unique index node_guid_uindex
    on node (guid);

create table model_template
(
    guid        integer not null
        constraint model_template_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table model_template
    owner to bioexplorer;

create unique index model_template_guid_uindex
    on model_template (guid);

create table node_type
(
    guid        integer not null
        constraint node_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table node_type
    owner to bioexplorer;

create unique index node_type_guid_uindex
    on node_type (guid);

create table model_type
(
    guid        integer not null
        constraint model_type_pk
            primary key,
    code        varchar not null,
    description integer
);

alter table model_type
    owner to bioexplorer;

create unique index model_type_guid_uindex
    on model_type (guid);

create table morphology_type
(
    guid        integer not null
        constraint morphology_type_pk
            primary key,
    code        varchar not null,
    description varchar
);

alter table morphology_type
    owner to bioexplorer;

create unique index morphology_type_guid_uindex
    on morphology_type (guid);

create table configuration
(
    guid  varchar not null
        constraint configuration_pk
            primary key,
    value varchar not null
);

alter table configuration
    owner to bioexplorer;

create unique index configuration_guid_uindex
    on configuration (guid);

create table section_type
(
    guid        integer not null
        constraint section_type_pk
            primary key,
    description varchar not null
);

alter table section_type
    owner to bioexplorer;

create unique index section_type_guid_uindex
    on section_type (guid);

create unique index section_type_description_uindex
    on section_type (description);

create table section
(
    morphology_guid     integer not null,
    section_guid        integer not null,
    section_parent_guid integer not null,
    section_type_guid   integer not null,
    points              bytea   not null
);

alter table section
    owner to bioexplorer;

create unique index section_morphology_guid_section_guid_uindex
    on section (morphology_guid, section_guid);

create table end_foot
(
    guid                   integer not null
        constraint end_foot_pk
            primary key,
    astrocyte_guid         integer not null,
    astrocyte_section_guid integer not null,
    vertices               bytea   not null,
    indices                bytea   not null
);

alter table end_foot
    owner to bioexplorer;

create unique index end_foot_guid_uindex
    on end_foot (guid);

